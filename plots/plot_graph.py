import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Set global font sizes for LaTeX-like appearance (from plot_accuracy.py)
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22, # plot_graph.py uses 22, plot_accuracy.py uses 24, let's use 22 to match plot_graph specific
    'figure.titlesize': 24
})

# Data from the table
#tokens = np.array([100, 200, 300, 400, 500, 600])
tokens = np.array([25, 50, 75, 150])
# Convert PPL to loss using log
no_shp_loss = np.array([2.15571, 1.42701, 1.25671,  1.12592])
shp_loss = np.array([1.66026, 1.31402, 1.23397, 1.13507])

baseline_loss =np.array([2.07070348, 2.01282085, 1.99077425, 1.97531637, 1.96365183,1.95407671])
mup_loss = np.array([2.05160775, 2.00272247, 1.97841883, 1.96564275, 1.95346724, 1.946053  ])
sambay_mup_loss = np.array([2.0242, 1.9766, 1.9548, 1.9385, 1.9311, 1.92245]) #old
sambayoco_mup_loss = np.array([2.0212, 1.9736, 1.9509, 1.9377, 1.9272, 1.9209]) #old
# sambay_mup_loss = np.array([2.0239, 1.9759, 1.9539, 1.9388, 1.9299, 1.9220]) #new
# sambayoco_mup_loss = np.array([2.0218, 1.9745, 1.9527, 1.9370, 1.9281, 1.9204]) #new
nobsscale_loss = np.array([2.05136352, 1.99953216, 1.97367817, 1.95834678, 1.94749461, 1.93951834])
mup_normal_loss = np.array([ 2.0966, 2.0283,1.996, 1.9770, 1.9617,1.95586])
super_mup_loss = np.array([2.0258, 1.9729, 1.9706, 1.9467, 1.9393,1.93159])
mup_wsd_loss = np.array([2.0301,1.9812,1.9604,1.9447,1.9373,1.9306])
super_mup_wsd_loss= np.array([2.0149,1.9720, 1.9542,1.9440,1.9359, 1.9407])
super_mup_wdsc_loss= np.array([2.0123,1.9778,1.9594,1.9519, 1.9432, 1.9602])
super_mup_wsd_wdsc_loss= np.array([2.0149,1.9753,1.9571,1.9462, 1.9600, 1.9570])
super_mup_lrsc_loss = np.array([2.0123,1.9682,1.9495,1.9365,1.9345, 1.9288])
super_mup_lrsc2_loss = np.array([2.0123,1.9692,1.9538,1.9428,np.nan,np.nan])
mup_lrsc_loss = np.array([2.0454, 1.9939, 1.9760,1.9612,1.9517,1.9466])
super_mup_lrsc_lecun_loss = np.array([2.0087,1.9649,1.9501,1.9399,np.nan,np.nan])
super_mup_lrsc_lecun_lr4_loss = np.array([2.0293, 1.9831, 1.9655, 1.9497, 1.9403,1.9330])
mup_untie_lecun_loss = np.array([2.0318,1.9818,1.9604,1.9425,1.9348,1.9272])
ori_mup_loss = np.array([2.0708,2.0162,1.9905,1.9698,1.9589,2.0075])
# super_mup_scale_loss = np.array([2.0172, 2.005, 2.002, 1.9467, 1.9393,np.nan])

# --- Plotting ---
plt.figure(figsize=(16, 12)) # Adjust figure size for better readability

# Define the power law function to fit: L = A * D^(-b) + C
def power_law_with_offset(D, A, b, C):
    return A * D**(-b) + C

plt.figure(figsize=(20, 12))  # Increased figure size for better readability

    # 'Transformer++ (SP)': baseline_loss,
    # 'Transformer++ (μP++)': nobsscale_loss,
# Data series to plot and fit
data_series = {
    # 'SP': baseline_loss,
    # 'μP': ori_mup_loss,
    # 'μP++': nobsscale_loss,
    # 'μP++ (Batch Scaling)': mup_loss,
    # 'μP++ (Normal Init.)': mup_normal_loss,

    # 'Transformer++ (SP)': baseline_loss,
    # 'Transformer++ (μP++)': nobsscale_loss,
    # 'Samba+YOCO (μP++)': sambayoco_mup_loss,
    # 'SambaY (μP++)': sambay_mup_loss,
    # 'Super μP++': super_mup_loss,
    # 'Super μP++ (wsd)': super_mup_wsd_loss,
    # 'Super μP++ (wdsc)': super_mup_wdsc_loss,
    # 'Super μP++ (wsd_wdsc)': super_mup_wsd_wdsc_loss,
    # 'Super μP++ (lrsc)': super_mup_lrsc_loss,
    # 'Super μP++ (lrsc2)': super_mup_lrsc2_loss,
    # #'Super μP++ (lrsc_lecun)': super_mup_lrsc_lecun_loss,
    'sesame': no_shp_loss,
    'sesame+shallotpeat:': shp_loss,
    # 'μP++ ': mup_untie_lecun_loss,   #untie lecun
    # 'μP++ (WSD)': mup_wsd_loss, #untie
    # # 'μP++ (LR Scaling + Normal Init.)': mup_lrsc_loss, #untie
    # 'μP++ (LR Scaling + Indep. WD)': super_mup_lrsc_lecun_lr4_loss, #untie lecun

}

markers = ['o','d', 's',  '^',  'v', '>', '<', 'p', '*', 'h', '8', 'D', 'P', 'X', '4', '3', '2']  # Added one more marker
colors = plt.cm.get_cmap('tab10', len(data_series))  # Using tab20 colormap for better color distinction

for i, (label, loss_data) in enumerate(data_series.items()):
    # Filter out NaNs for both tokens and loss_data
    valid_mask = ~np.isnan(loss_data)
    current_tokens = tokens[valid_mask]
    current_loss = loss_data[valid_mask]

    if len(current_tokens) > 1: # Need at least 2 points to fit a line label=label,
        # Plot original data points as markers only
        plt.plot(current_tokens, current_loss, marker=markers[i % len(markers)], linestyle='none', label=label, color=colors(i),markersize=16,)

        # Initial guesses for parameters A, b, C
        # These are heuristics and might need tuning if fitting fails
        c_guess = np.min(current_loss) * 0.95 # Asymptotic loss slightly below min observed
        b_guess = 0.1 # Small positive exponent
        # Ensure c_guess is less than the first loss point for a_guess calculation
        if c_guess >= current_loss[0]:
            c_guess_safe = current_loss[0] * 0.95
        else:
            c_guess_safe = c_guess
        a_guess = (current_loss[0] - c_guess_safe) * (current_tokens[0]**b_guess)
        if a_guess <= 0: # A should ideally be positive
            a_guess = 1.0

        initial_params = [a_guess, b_guess, c_guess]
        param_bounds = ([0, 0, 0], [np.inf, np.inf, np.min(current_loss)]) # Bounds: A>0, b>0, 0 < C < min_loss

        try:
            params, covariance = curve_fit(power_law_with_offset, current_tokens, current_loss, 
                                           p0=initial_params, bounds=param_bounds, maxfev=5000)
            A, b, C_fit = params
            
            # Calculate R-squared for goodness of fit
            fitted_loss_at_data_points = power_law_with_offset(current_tokens, A, b, C_fit)
            ss_res = np.sum((current_loss - fitted_loss_at_data_points)**2)
            ss_tot = np.sum((current_loss - np.mean(current_loss))**2)
            if ss_tot == 0: # Avoid division by zero if all y values are the same
                r_squared = 1.0 if ss_res == 0 else 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Generate y-values for the fitted line
            # Ensure tokens for plotting are sorted for a smooth line, especially if original tokens aren't.
            plot_tokens = np.geomspace(current_tokens.min(), current_tokens.max(), 100)
            fitted_loss_for_plotting = power_law_with_offset(plot_tokens, A, b, C_fit)
            
            # Plot the fitted line
            plt.plot(plot_tokens, fitted_loss_for_plotting, linestyle='--', color=colors(i), label=f'{label} (A={A:.2f},b={b:.2f},C={C_fit:.2f})',linewidth=3)
        except RuntimeError:
            print(f"Could not fit {label} to power law with offset.")
            # Optionally, plot just the points if fit fails, or skip plotting fit
            pass # Currently, just prints a message and skips fit line

    elif len(current_tokens) == 1:
        # If only one point, just plot the point (as marker)
        plt.plot(current_tokens, current_loss, marker=markers[i % len(markers)], linestyle='none', label=label, color=colors(i))


# Plot on log-log scale, handling potential NaNs
# plt.plot(tokens[~np.isnan(baseline_loss)], baseline_loss[~np.isnan(baseline_loss)], marker='o', linestyle='-', label='SP')
# #plt.plot(tokens[~np.isnan(mup_loss)], mup_loss[~np.isnan(mup_loss)], marker='s', linestyle='-', label='μP++&batchsize_scaling')
# plt.plot(tokens[~np.isnan(nobsscale_loss)], nobsscale_loss[~np.isnan(nobsscale_loss)], marker='^', linestyle='-', label='μP++')
# plt.plot(tokens[~np.isnan(sambay_mup_loss)], sambay_mup_loss[~np.isnan(sambay_mup_loss)], marker='d', linestyle='-', label='μP++ (sambay)')
# plt.plot(tokens[~np.isnan(sambayoco_mup_loss)], sambayoco_mup_loss[~np.isnan(sambayoco_mup_loss)], marker='d', linestyle='-', label='μP++ (sambayoco)')
# #plt.plot(tokens[~np.isnan(mup_normal_loss)], mup_normal_loss[~np.isnan(mup_normal_loss)], marker='*', linestyle='-', label='μP++&normal0.02')
# # plt.plot(tokens[~np.isnan(mup_lrsc_loss)], mup_lrsc_loss[~np.isnan(mup_lrsc_loss)], marker='d', linestyle='-', label='μP++ lrsc non-independent wd (untie)')
# # plt.plot(tokens[~np.isnan(super_mup_loss)], super_mup_loss[~np.isnan(super_mup_loss)], marker='d', linestyle='-', label='Super μP++ (untie, lecun uniform, zeroinit)')
# #plt.plot(tokens[~np.isnan(mup_wsd_loss)], mup_wsd_loss[~np.isnan(mup_wsd_loss)], marker='d', linestyle='-', label='μP++ wsd (untie)')
# #plt.plot(tokens[~np.isnan(super_mup_wdsc_loss)], super_mup_wdsc_loss[~np.isnan(super_mup_wdsc_loss)], marker='d', linestyle='-', label='Super μP++ wdsc (untie)') #lr_linear, untie, zeroinit, embed1e-4
# #plt.plot(tokens[~np.isnan(super_mup_wsd_loss)], super_mup_wsd_loss[~np.isnan(super_mup_wsd_loss)], marker='d', linestyle='-', label='Super μP++ wsd (untie)')
# #plt.plot(tokens[~np.isnan(super_mup_wsd_wdsc_loss)], super_mup_wsd_wdsc_loss[~np.isnan(super_mup_wsd_wdsc_loss)], marker='d', linestyle='-', label='Super μP++ wsd_wdsc (untie)')
# #plt.plot(tokens[~np.isnan(super_mup_lrsc_lecun_loss)], super_mup_lrsc_lecun_loss[~np.isnan(super_mup_lrsc_lecun_loss)], marker='d', linestyle='-', label='Super μP++ lrsc (untie, lecun uniform)')
# # plt.plot(tokens[~np.isnan(super_mup_lrsc_loss)], super_mup_lrsc_loss[~np.isnan(super_mup_lrsc_loss)], marker='d', linestyle='-', label='Super μP++ lrsc (untie)')
# # plt.plot(tokens[~np.isnan(super_mup_lrsc_lecun_lr4_loss)], super_mup_lrsc_lecun_lr4_loss[~np.isnan(super_mup_lrsc_lecun_lr4_loss)], marker='d', linestyle='-', label='μP++ lrsc independent wd (untie, lecun uniform)')
# #plt.plot(tokens[~np.isnan(super_mup_lrsc2_loss)], super_mup_lrsc2_loss[~np.isnan(super_mup_lrsc2_loss)], marker='d', linestyle='-', label='Super μP++ lrsc non-independent wd (untie)')
plt.xscale('log')   
# plt.yscale('log') # Y-axis should be linear for L = A*D^(-b) + C

plt.xlabel("Training Tokens (Billions)", fontsize=24)
plt.ylabel("Validation Loss", fontsize=24)
plt.legend()#bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)  # Moved legend outside
plt.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate external legend
plt.savefig('scaling_data_1B_mup_abl_untie.png', dpi=300, bbox_inches='tight')
plt.show()
