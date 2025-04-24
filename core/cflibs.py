
"""
Core functions for Calibration-Free LIBS (CF-LIBS) calculations.
Includes Boltzmann plot for temperature, Saha-Boltzmann for electron density,
and CF-LIBS concentration estimation.

Note: Saha-Boltzmann and CF-LIBS implementations are simplified approximations
      and require careful validation and potentially more rigorous methods
      (e.g., proper handling of partition functions, optical thickness checks)
      for accurate quantitative results. Requires valid atomic data files.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set

# --- SciPy Imports ---
try: from scipy.stats import linregress; SCIPY_AVAILABLE = True
except ImportError: SCIPY_AVAILABLE = False; logging.error("SciPy not found. CF-LIBS unavailable."); 
def linregress(*a,**k): raise ImportError("SciPy required.")

# Import data models and atomic data functions
from .data_models import Peak, NISTMatch
from .atomic_data import get_partition_function, get_ionization_energy

# --- Physical Constants ---
K_B_EV = 8.617333262e-5 # Boltzmann constant in eV/K
# Constants for Saha (SI units)
H_EV_S = 4.135667696e-15; M_E_KG = 9.1093837015e-31
K_B_J = K_B_EV * 1.602176634e-19; H_J_S = H_EV_S * 1.602176634e-19
SAHA_FACTOR_SI = 2 * ( (2 * np.pi * M_E_KG * K_B_J) / (H_J_S**2) )**(3/2) # m^-3 K^(-3/2)

# --- Boltzmann Plot Calculation ---
def calculate_boltzmann_temp(lines_data: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[pd.DataFrame]]:
    """ Calculates plasma temperature using the Boltzmann plot method. """
    if not SCIPY_AVAILABLE: logging.error("SciPy unavailable for Boltzmann."); return None,None,None,None
    required=['intensity','wavelength_nm','ei_upper','gi_upper','aki']; missing=[c for c in required if c not in lines_data.columns];
    if missing: logging.error(f"Boltzmann failed: Missing {missing}"); return None,None,None,None
    try: # Data Prep
        df=lines_data[required].copy(); [df[c].__setitem__(pd.to_numeric(df[c],errors='coerce')) for c in required]; ir=len(df); df.dropna(subset=required,inplace=True); crit_pos=['intensity','wavelength_nm','gi_upper','aki']; [df.__setitem__(df[df[c]>1e-12]) for c in crit_pos]; df=df[np.isfinite(df['ei_upper'])]; fr=len(df);
        if fr<ir: logging.warning(f"Dropped {ir-fr} invalid rows for Boltzmann.")
        if fr<2: logging.error(f"Need >=2 valid pts for Boltzmann, found {fr}."); return None,None,None,None
        df['x_energy_ev']=df['ei_upper']; num=df['intensity']*df['wavelength_nm']; den=df['aki']*df['gi_upper']; mask=(den>1e-12)&(num>1e-12);
        if not mask.all(): logging.warning(f"Removing {len(df)-mask.sum()} rows non-positive log term."); df=df[mask]
        if len(df)<2: logging.error("Not enough pts after log check."); return None,None,None,None
        # Calculate y = ln( I * lambda / (A_ki * g_k) ) --- Note: NIST gk = Upper state g
        df['y_boltzmann_term']=np.log(num[mask]/den[mask]);
        if 'label' in lines_data.columns: df['label']=lines_data.loc[df.index,'label'] # Keep label if present
        x,y=df['x_energy_ev'].to_numpy(),df['y_boltzmann_term'].to_numpy();
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)): logging.error("NaN/Inf Boltzmann coords."); return None,None,None,df
    except Exception as e: logging.error(f"Boltzmann data prep error:{e}",exc_info=True); return None,None,None,None
    try: # Regression
        slope, intercept, r_val, p_val, stderr = linregress(x,y); r2 = r_val**2; logging.info(f"Boltzmann fit: Slope={slope:.4f}, R²={r2:.4f}, StdErr={stderr:.4e}")
        # Slope = -1 / (k_B * T_e)
        if slope>=-1e-9: logging.warning(f"Boltzmann slope non-negative ({slope:.4f}). Temp invalid."); return None,None,r2,df
        temp = -1.0/(slope*K_B_EV); temp_err=abs((1.0/(slope**2*K_B_EV))*stderr) if np.isfinite(stderr) and slope!=0 else np.nan; logging.info(f"Calc Temp:{temp:.1f} +/- {temp_err:.1f} K"); return temp, temp_err, r2, df[['x_energy_ev', 'y_boltzmann_term', 'label'] if 'label' in df else ['x_energy_ev', 'y_boltzmann_term']] # Return plot data
    except Exception as e: logging.error(f"Boltzmann fit error: {e}",exc_info=True); return None,None,None,df[['x_energy_ev', 'y_boltzmann_term', 'label'] if 'label' in df else ['x_energy_ev', 'y_boltzmann_term']]

# --- Saha-Boltzmann for Electron Density ---
def calculate_electron_density_saha(lines_ion1: pd.DataFrame, lines_ion2: pd.DataFrame,
                                    temperature_k: float, species1_key: str, species2_key: str) -> Optional[float]:
    """ Estimates electron density (Ne) using the Saha-Boltzmann equation ratio method. """
    logging.info(f"Attempting Saha Nₑ calc: {species1_key}/{species2_key} @ T={temperature_k:.0f}K.")
    if lines_ion1.empty or lines_ion2.empty: logging.error("Saha: Input DFs empty."); return None
    if not np.isfinite(temperature_k) or temperature_k <= 0: logging.error("Saha: Invalid Tₑ."); return None

    U1 = get_partition_function(species1_key, temperature_k)
    U2 = get_partition_function(species2_key, temperature_k)
    V_ion = get_ionization_energy(species1_key)
    if None in [U1, U2, V_ion] or U1 <= 0 or U2 <= 0 or V_ion <= 0: logging.error(f"Saha: Missing atomic data (U1={U1}, U2={U2}, Vion={V_ion})."); return None

    required=['intensity','aki','gi_upper','ei_upper']; # Using upper state g_k, E_k
    def clean(df,sp): # Inner helper to clean data
        if not all(c in df.columns for c in required): logging.error(f"Saha: Missing cols for {sp}."); return pd.DataFrame()
        df_c=df[required].copy(); [df_c[c].__setitem__(pd.to_numeric(df_c[c], errors='coerce')) for c in required]; df_c.dropna(inplace=True);
        for col in ['intensity','aki','gi_upper']: df_c=df_c[df_c[col]>1e-12] # Must be positive
        df_c=df_c[np.isfinite(df_c['ei_upper'])] # Energy can be zero? Yes.
        return df_c
    df1=clean(lines_ion1,species1_key); df2=clean(lines_ion2,species2_key);
    if df1.empty or df2.empty: logging.error(f"Saha: No valid lines after cleaning for {species1_key} or {species2_key}."); return None

    kbt_ev = K_B_EV * temperature_k
    try:
        # Calculate term = ln(I / (A_ki * g_k)) + E_k / (k_B * T) for each line
        # This term is proportional to ln(N_s / U_s) + constant
        df1['log_term'] = np.log(df1['intensity'] / (df1['aki'] * df1['gi_upper'])) + df1['ei_upper'] / kbt_ev
        df2['log_term'] = np.log(df2['intensity'] / (df2['aki'] * df2['gi_upper'])) + df2['ei_upper'] / kbt_ev

        # Average these terms (use median for robustness against outliers?)
        avg_log1 = df1['log_term'].median()
        avg_log2 = df2['log_term'].median()
        if not np.isfinite(avg_log1) or not np.isfinite(avg_log2): logging.error("Saha: NaN/Inf avg log terms."); return None

        # Solve for ln(Ne) using the log form of Saha-Boltzmann:
        # Ref: e.g., Cristoforetti et al., Spectrochimica Acta Part B 65 (2010) 86–95, Eq. (10)
        # ln(Iλ/Ag)_1 - ln(Iλ/Ag)_2 + (E1-E2)/kT = ln(Ne) + ln(2*U1/U2) - ln(SAHA_FACTOR_SI * T^1.5) + Vion/kT ??? -> Complex rearrangement needed.
        # Alternative formulation often used: Plot ln(Iλ/Ag)_corr vs E. Slope gives T. Intercept difference gives Ne.
        # Let's use the formula relating the average terms directly, derived from Saha equation (needs validation):
        # ln(Ne [m^-3]) = (avg_log1 - avg_log2) + ln(2*U2/U1) + ln(SAHA_FACTOR_SI) + 1.5*ln(T [K]) - (V_ion [eV] / kbt_ev)
        # Note the factor of 2 for the electron partition function g_e=2
        ln_Ne = (avg_log1 - avg_log2) + np.log(2 * U2 / U1) + np.log(SAHA_FACTOR_SI) + 1.5 * np.log(temperature_k) - (V_ion / kbt_ev)

        Ne_m3 = np.exp(ln_Ne) # Electron density in m^-3
        Ne_cm3 = Ne_m3 * 1e-6 # Convert to cm^-3

        if not np.isfinite(Ne_cm3) or Ne_cm3 <= 0: logging.error(f"Saha: Invalid Ne calc ({Ne_cm3:.3e}). Check input lines/data."); return None
        logging.info(f"Estimated Nₑ (Saha Approx.): {Ne_cm3:.3e} cm⁻³ for {species1_key}/{species2_key}")
        return Ne_cm3
    except Exception as e: logging.error(f"Saha calculation error: {e}", exc_info=True); return None


# --- CF-LIBS Concentration ---
def _filter_lines_for_cflibs(peaks: List[Peak], max_delta_lambda_nm: float = 0.05, min_fit_r2: float = 0.90) -> pd.DataFrame:
    """ Filters peaks to select usable lines for CF-LIBS calculation. """
    # ... (Function remains same as Part 30) ...
    usable=[]; required=['ei','gi','aki'] # NISTMatch attrs (ei=Ek, gi=gk)
    for peak in peaks:
        if not peak.best_fit or not peak.best_fit.success: continue
        if peak.best_fit.r_squared is not None and peak.best_fit.r_squared<min_fit_r2: continue
        if not peak.potential_matches: continue
        best_match=None
        for match in peak.potential_matches:
            if all(getattr(match,a,None) is not None and np.isfinite(getattr(match,a)) for a in required if a!='ei') \
            and getattr(match,'ei',None) is not None and np.isfinite(getattr(match,'ei')):
                 if getattr(match,'aki',0)>1e-12 and getattr(match,'gi',0)>1e-12: best_match=match; break
        if not best_match: continue
        wl_diff=abs(peak.wavelength_fitted_or_detected - best_match.wavelength_db)
        if wl_diff > max_delta_lambda_nm: continue
        # TODO: Add Optically Thin Check Placeholder/Warning
        usable.append({'element':best_match.element, 'species':f"{best_match.element} {best_match.ion_state_str}", 'ion_stage':best_match.ion_state_int, 'wavelength_nm':peak.wavelength_fitted_or_detected, 'intensity':peak.best_fit.amplitude, 'aki':best_match.aki, 'ei_upper':best_match.ei, 'gi_upper':best_match.gi})
    if not usable: logging.warning("CF-LIBS Filter: No usable lines found."); return pd.DataFrame()
    return pd.DataFrame(usable)


def calculate_cf_libs_conc(peaks: List[Peak], temperature_k: float,
                           electron_density_cm3: Optional[float] = None, # Currently unused but could be incorporated
                           max_delta_lambda_nm: float = 0.05, min_fit_r2: float = 0.90
                           ) -> Optional[pd.DataFrame]:
    """ Estimates elemental concentrations using the CF-LIBS method (simplified Boltzmann/Normalization). """
    logging.info(f"Attempting CF-LIBS Conc calc @ T={temperature_k:.0f}K.")
    if not peaks: logging.error("CF-LIBS: No peaks."); return None
    if not np.isfinite(temperature_k) or temperature_k <= 0: logging.error("CF-LIBS: Invalid Tₑ."); return None

    usable_df=_filter_lines_for_cflibs(peaks, max_delta_lambda_nm, min_fit_r2);
    if usable_df.empty: logging.error("CF-LIBS: No usable lines found."); return None
    logging.info(f"CF-LIBS: Using {len(usable_df)} filtered lines.")

    all_species=set(usable_df['species']); partition_funcs:Dict[str,Optional[float]]={}; missing_U=False
    for sp in all_species: U=get_partition_function(sp, temperature_k);
    if U is None: logging.error(f"CF-LIBS: Missing U(T) for {sp}."); missing_U=True; partition_funcs[sp]=U
    if missing_U: logging.error("CF-LIBS cannot proceed without U(T)."); return None

    kbt_ev=K_B_EV*temperature_k;
    try: # Calculate F value: ln(I*lambda / (A*g)) + E_k / kBT
        usable_df['F_value'] = (np.log(usable_df['intensity'] * usable_df['wavelength_nm'] / (usable_df['aki'] * usable_df['gi_upper']))
                               + usable_df['ei_upper'] / kbt_ev)
        usable_df.dropna(subset=['F_value'], inplace=True)
        if usable_df.empty: logging.error("CF-LIBS: No valid F values calculated."); return None
    except Exception as e: logging.error(f"CF-LIBS: Error calculating F values: {e}",exc_info=True); return None

    # Calculate average F_s_avg + ln(U_s) = Q_s ≈ ln(C_s * N_total * Factor)
    species_Q:Dict[str,float]={}; species_lines_count: Dict[str, int] = {}
    for sp in all_species:
        lines=usable_df[usable_df['species']==sp]; F_vals=lines['F_value'].dropna();
        if len(F_vals)==0: logging.warning(f"CF-LIBS: No valid F for {sp}."); continue
        F_avg=F_vals.mean(); U_s=partition_funcs.get(sp); # Use mean F value
        if U_s is None or U_s<=0: continue # Should have been checked
        Q_s=F_avg+np.log(U_s);
        if not np.isfinite(Q_s): logging.warning(f"CF-LIBS: Invalid Q for {sp}."); continue
        species_Q[sp]=Q_s; species_lines_count[sp] = len(F_vals)
        logging.debug(f"CF-LIBS: {sp}, Lines={len(F_vals)}, F_avg={F_avg:.3f}, U={U_s:.3f}, Q_s={Q_s:.3f}")

    if not species_Q: logging.error("CF-LIBS: Failed calc Q for any species."); return None

    # Estimate concentrations C_s ≈ exp(Q_s) / Sum[exp(Q_s)]
    exp_Q={sp:np.exp(Q) for sp,Q in species_Q.items()}; total_exp_Q=sum(exp_Q.values())
    if total_exp_Q<=0 or not np.isfinite(total_exp_Q): logging.error(f"CF-LIBS: Invalid total_exp_Q ({total_exp_Q})."); return None

    element_conc:Dict[str,float]={};
    for sp,exp_Q_val in exp_Q.items():
        element = sp.split()[0]; conc_sp = exp_Q_val / total_exp_Q # Conc of this *species* relative to total *intensity factor*
        element_conc[element] = element_conc.get(element, 0.0) + conc_sp # Sum contributions for element

    # Normalize elemental concentrations to 1 (or 100%)
    total_conc_sum = sum(element_conc.values())
    if total_conc_sum <=0 or not np.isfinite(total_conc_sum): logging.error("CF-LIBS: Invalid total element concentration sum."); return None

    results = [{'Element':el, 'Concentration':conc / total_conc_sum} for el,conc in element_conc.items()]
    df = pd.DataFrame(results).sort_values('Concentration', ascending=False)

    logging.info(f"CF-LIBS Concentration Result:\n{df.round(4)}")
    return df