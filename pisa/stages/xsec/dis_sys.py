"""
Stage to apply pre-calculated Genie uncertainties
"""

from __future__ import absolute_import, print_function, division

__all__ = ["dis_sys", "SIGNATURE", "apply_dis_sys"]

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.profiler import profile
from pisa.utils.fileio import from_file
from pisa.utils.numba_tools import WHERE


class dis_sys(PiStage): # pylint: disable=invalid-name
    """
    Stage to apply pre-calculated Genie systematics.

    Parameters
    ----------
    data
    params
        Must contain ::
            dis_csms : quantity (dimensionless)

        extrapolation_type : string
            choice of ['constant', 'linear', 'higher']

    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

    Notes
    -----
    Requires the events have the following keys ::

        bjorken_y
            Inelasticity
        dis
            1 if event is DIS, else 0

    """
    def __init__(
        self,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
        extrapolation_type='constant',
    ):
        expected_params = (
            'DIS',
        )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = (
            'xsec_csms',
        )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = (
            'weights',
        )

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode is not None
        assert self.calc_mode is None
        assert self.output_mode is not None

        self.extrapolation_type = extrapolation_type

    @profile
    def setup_function(self):

        extrap_dict = from_file('tot_xsec_corr_Q2min1_isoscalar.pckl')

        # load splines
        wf_nucc =    from_file('NuMu_CC_flat.pckl')
        wf_nubarcc = from_file('NuMu_Bar_CC_flat.pckl')
        wf_nunc =    from_file('NuMu_NC_flat.pckl')
        wf_nubarnc = from_file('NuMu_Bar_NC_flat.pckl')


        # TODO: only for DIS events!!

        
        for container in self.data:

            # creat keys for external dict

            if container.name.endswith('_cc'):
                current = 'CC'
            elif container.name.endswith('_nc'):
                current = 'NC'
            else:
                raise ValueError('Can not determine whether container with name "%s" is pf type CC or NC based on its name'%container.name)
            nu = 'Nu' if container['nubar'] > 0 else 'NuBar'

            lgE = np.log10(container['true_energy'].get('host'))
            bjorken_y = container['bjorken_y'].get('host')

            #
            # Calculate variation of total cross section
            #

            lgE_min = 2. if self.extrapolation_type == 'constant' else 1.68
            mask = lgE <= lgE_min

            w_tot = np.ones_like(lgE)
            
            poly_coef = extrap_dict[nu][current]['poly_coef']
            w_tot[~mask] = np.polyval(poly_coef, lgE[~mask])
           
            if self.extrapolation_type == 'constant':
                poly_coef = extrap_dict[nu][current]['poly_coef']
                w_tot[mask] = np.polyval(poly_coef, lgE_min * w_tot[mask]) 
            elif self.extrapolation_type == 'linear':
                lin_coef = extrap_dict[nu][current]['linear']
                w_tot[mask] = np.polyval(lin_coef, lgE[mask])
            elif self.extrapolation_type == 'higher':
                poly_coef = extrap_dict[nu][current]['poly_coef']
                w_tot[mask] = np.polyval(poly_coef, lgE[mask]) 
            else:
                raise ValueError('Unknown extrapolation type "%s"'%self.extrapolation_type)
          
            container["dis_correction_total"] = w_tot

            #
            # Calculate variation of differential cross section
            #

            lgE_min = 2.
            mask = lgE <= lgE_min

            w_diff = np.ones_like(lgE)

            if current == 'CC' and container['nubar'] > 0:
                weight_func = wf_nucc
            elif current == 'CC' and container['nubar'] < 0:
                weight_func = wf_nubarcc
            elif current == 'NC' and container['nubar'] > 0:
                weight_func = wf_nunc
            elif current == 'NC' and container['nubar'] < 0:
                weight_func = wf_nubarnc

            w_diff[~mask] = weight_func.ev(lgE[~mask], bjorken_y[~mask])
            w_diff[mask] = weight_func.ev(w_diff[mask] * lgE_min, bjorken_y[mask])
            
            container["dis_correction_diff"] = w_diff
         

    @profile
    def apply_function(self):
        dis_ma_qe = self.params.Genie_Ma_QE.m_as('dimensionless')
        dis_ma_res = self.params.Genie_Ma_RES.m_as('dimensionless')

        for container in self.data:
            apply_dis_sys(
                dis_ma_qe,
                container['linear_fit_maccqe'].get(WHERE),
                container['quad_fit_maccqe'].get(WHERE),
                dis_ma_res,
                container['linear_fit_maccres'].get(WHERE),
                container['quad_fit_maccres'].get(WHERE),
                out=container['weights'].get(WHERE),
            )
            container['weights'].mark_changed(WHERE)


if FTYPE == np.float64:
    SIGNATURE = '(f8, f8, f8, f8, f8, f8, f8[:])'
else:
    SIGNATURE = '(f4, f4, f4, f4, f4, f4, f4[:])'
@guvectorize([SIGNATURE], '(),(),(),(),(),()->()', target=TARGET)
def apply_dis_sys(
    dis_ma_qe,
    linear_fit_maccqe,
    quad_fit_maccqe,
    dis_ma_res,
    linear_fit_maccres,
    quad_fit_maccres,
    out,
):
    out[0] *= (
        (1. + (linear_fit_maccqe + quad_fit_maccqe * dis_ma_qe) * dis_ma_qe)
        * (1. + (linear_fit_maccres + quad_fit_maccres * dis_ma_res) * dis_ma_res)
    )
