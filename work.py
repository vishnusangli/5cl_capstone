# %%
import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backend
import math
import tqdm
import scipy.optimize as sc
import scipy.signal as sig
#%%
dir = "spect_1"
df_pow = pd.read_csv(f"{dir}/pow.csv", dtype=np.float64)
df_lights = []
for i in range(7):
    temp = pd.read_csv(f"{dir}/v{i}.csv", dtype = np.float64())
    df_lights.append(temp)
# %%
plt.figure(figsize = (10, 8))
for i in range(7):
    temp = df_lights[i]
    plt.plot(temp["Wavelength"], temp["Count"], label = f"{df_pow.V[i]} V, {df_pow.mA[i]} mA")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Count")
plt.title("Blackbody Spectrum of an LED")
plt.legend()
plt.grid()
plt.savefig("exp1_1.png")
# %% 
### Removing Background
for i in range(7):
    temp = df_lights[i]
    temp["Count"] = temp["Count"] - df_lights[0]["Count"]
    temp["lambda"] = temp["Wavelength"] / np.power(10, 9)

# %%
## Constants
C = np.float64(299792458)
plank_const = np.float64(6.62607015 * 1e-34)
boltzmann_const = np.float64(1.3807 * 1e-23)
light_speed = C
T = np.float64(6000)

freq_p0 = [np.divide(2 * plank_const, np.power(light_speed, 3)), np.divide(plank_const, boltzmann_const * T)]
wave_p0 = [ 8 * np.pi * plank_const * light_speed, np.divide(plank_const * light_speed, boltzmann_const * T)]
wave_t_part = np.divide(plank_const * light_speed, boltzmann_const)

def gen_wave_p0(t, c, d):
    return [ 8 * np.pi * plank_const * light_speed, np.divide(plank_const * light_speed, boltzmann_const * t), c, d]

def gen_freq_p0(t, c, d):
    return [np.divide(2 * plank_const, light_speed**2), np.divide(plank_const, boltzmann_const * t), c, d]
# %%
## Converting Frequency
def w_f(x):
    """
    convert wavelength to frequency
    """
    return np.divide(C, x)


for i in range(7):
    temp = df_lights[i]
    temp["Frequency"] = w_f(temp["lambda"])
# %%
## Plot Frequency
plt.figure(figsize = (10, 8))
for i in range(7):
    temp = df_lights[i]
    plt.plot(temp["Frequency"], temp["Count"], label = f"{df_pow.V[i]} V, {df_pow.mA[i]} mA")

plt.xlabel("Frequency")
plt.ylabel("Count")
plt.title("Blackbody Spectrum of an LED")
plt.legend()
plt.grid()
# %%
##Defining Functions
def frequency_func(x, a, b, c, d):
    val1 = a * np.power((x + c), 3)
    val2 = np.exp(b * (x + c)) - 1
    return np.divide(val1, val2) + d

def wave_func(x, a, b, c, d):
    val1 = np.divide(a,  np.power((x + d), 5))
    val2 = np.exp(np.divide(b, (x+d))) -1
    return np.divide(val1, val2) + c

def real_freq_func(x, T=3000):
    val1 = 8 * np.pi * plank_const
    val2 = np.power(np.divide(x, light_speed), 3)
    val3 = np.exp(np.divide(plank_const * x, boltzmann_const * T)) - 1
    return np.divide(val1 * val2, val3)

def real_wave_func(x, T=3000):
    val1 = np.divide(8 * np.pi * plank_const * light_speed, np.power(x, 5))
    val2 = np.exp(plank_const * light_speed, x * T * boltzmann_const) - 1
    return np.divide(val1, val2)

def raylegh_jeans_freq(x, t):
    val1 = 8 * np.pi * np.power(x, 2) * plank_const * T
    val2 = np.power(light_speed, 3)
    return np.divide(val1, val2)

def rj_func(x, a, b):
    val1 = a * np.poower(x, 2)
    return val1 + b


def from_wave_p0(p0):
    p0_vals = [i[0] for i in p0]
    diff0 = np.divide(wave_p0[0], p0_vals[0])
    val2 = np.divide(wave_t_part, p0_vals[1])
    return diff0, val2
# %%
# ##Performing Wavelength fit
min_w = 400
max_w = 800
def in_range(x):
    return x >= min_w and x <= max_w
in_range = np.vectorize(in_range)
# %%
funcs = []
wave_popts = []
factor = 1000
plt.figure(figsize = (10, 8))
col = "Smooth"
type_curve = "lambda"
for i in range(1, len(df_lights)):
    print(f"{df_pow.V[i]}")
    temp = df_lights[i]
    temp2 = temp[in_range(temp.Wavelength)]
    yfit, f, t = backend.weighted_fit(temp2[type_curve], factor * temp2[col], wave_func, p0 = gen_wave_p0(5000, 1000, 0), maxfev = 5000)
    wave_popts.append(t)
    funcs.append(f)
    plt.plot(temp[type_curve], temp[col], label = f"{df_wien.V[i]} V")
    plt.plot(temp[type_curve], f(temp[type_curve])/factor, linestyle = 'dashed')
    plt.xlabel("Wavelength (m)")
    plt.ylabel(f"Count")
    plt.title("Blackbody Spectrum Fits")
    plt.legend()
    #plt.yscale("log")
    plt.grid()
    plt.savefig("exp2_3.png")
plt.grid()
plt.savefig("exp2_3.png")

# %%
## Planks constant
a_vals = [i[0] + i[1] for i in wave_popts]

a_vals = pd.DataFrame(a_vals, columns = ["a", "a_unc", "b", "b_unc"])
a_vals["V"] = list(df_wien["V"][1:])
a_vals["Temp"] = list(df_wien["Temp"][1:])

# %%
a_vals["h"] = np.divide(a_vals["a"], 2 * light_speed)
a_vals["h_unc"] = np.divide(a_vals["a_unc"], 2 * light_speed)
rat = np.divide(a_vals.h[2], plank_const)
a_vals["h"] = a_vals["h"]/rat
a_vals["h_unc"] = a_vals["h_unc"]/rat

#a_vals["factor"] = 100* np.divide(abs(a_vals["h"] - plank_const), plank_const)
agrement = np.divide(abs(plank_const - a_vals.h), a_vals.h_unc) 
a_vals["agreement"] = round(agrement, 2)
a_vals["agree"] = a_vals["agreement"] <= 1
a_vals.loc[:, ["V", "a", "h", "h_unc", "agreement", "agree"]]
# %%
## Performing Frequency Fit
for i in range(7):
    print(f"{df_pow.V[i]}")
    temp = df_lights[i]
    yfit, f, t = backend.weighted_fit(temp["Frequency"], temp["Count"], frequency_func, p0 = freq_p0)
    plt.plot(temp["Frequency"], temp["Count"], label = "Data")
    plt.plot(temp["Frequency"], yfit, label = "fit", linestyle = 'dashed')
    plt.show()

## Plotting Example vals
for i in range(7):
    print(f"{df_pow.V[i]}")
    temp = df_lights[i]
    plt.plot(temp["Frequency"], temp["Count"], label = "Data")
    plt.plot(temp["Frequency"], real_freq_func(temp["Frequency"], T = 4000), linestyle = 'dashed')
    plt.show()
# %%
## Plotting Example Wave vals
plt.figure(figsize = (10, 8))
t = 5600
div = 2 * 10e2
intensity = []
for i in df_lights[0]["lambda"]:
    intensity.append(wave_func(i, *gen_wave_p0(t, 0, 0)))
intensity = np.divide(np.array(intensity), div)
plt.plot(df_lights[0]["lambda"], intensity, color = "black", linestyle = 'dashed', label="fit")
for i in range(7):
    temp = df_lights[i]
    plt.plot(temp["lambda"], temp["Count"], label = f"{df_pow.V[i]} V, {df_pow.mA[i]} mA")
plt.xlabel("Wavelength (m)")
plt.ylabel("Count")
plt.title("Blackbody Spectrum of an LED")
plt.legend()
plt.grid()

# %%
### SMOOTHEN DATA
plt.figure(figsize = (10, 8))
for i in range(0, 7):
    temp = df_lights[i]
    smoothen = sig.savgol_filter(temp["Count"], window_length=101, polyorder=3)
    plt.plot(temp["lambda"], temp["Count"], label = "Data")
    temp["Smooth"] = smoothen
    plt.plot(temp["lambda"], smoothen, linestyle = 'dashed', label = "smoothen")
    #yfit, f, t = backend.weighted_fit(temp["lambda"], smoothen, wave_func, p0 = [ 5.06490859e-28,  1.72526577e-06, -2.64867021e+01, -3.49999960e-07] )
    #plt.plot(temp["lambda"], yfit, linestyle = 'dashed', label = "Fit")
    #print(from_wave_p0(t))
# %%
## Wien's Displacement
sec = []
for temp in df_lights:
    sec.append(np.array(temp[temp.Smooth == max(temp.Smooth)])[0])
sec = np.array(sec)
first = np.array(df_pow)

df_wien = pd.DataFrame(np.column_stack([first, sec]), columns = ["V", "mA"] + list(df_lights[0].columns))
# %%
plt.figure(figsize = (10, 8))
for i in range(7):
    temp = df_lights[i]
    plt.plot(temp["Wavelength"], temp["Smooth"], label = f"{df_pow.V[i]} V, {df_pow.mA[i]} mA")

plt.xlabel("Wavelength (m)")
plt.ylabel("Count")
plt.title("Smoothened Blackbody Spectrum of an LED")
plt.legend()
plt.grid()
plt.savefig("exp1_2.png")
# %%
df_wien["Temp"] = (2.898 * 1e-3)/df_wien["lambda"]
# %%
plt.figure(figsize = (10, 8))
plt.scatter(df_wien["V"][1:], df_wien["Temp"][1:])
plt.plot(df_wien["V"][1:], df_wien["Temp"][1:], color = "orange", linestyle = "--")
plt.ylabel("Temperature (K)")
plt.xlabel("Voltage (V)")
plt.title("Temperature over Voltage")
plt.xticks(range(3, 14))
plt.legend()
plt.grid()
plt.savefig("exp1_3.png")
# %%

# %%
alpha = 4.5 * 1e-3
R_0 , T_0 = 161, 294
def det_temp(r):
    val = np.divide(np.divide(r, R_0) - 1, alpha)
    return val + T_0

 

df_wien["R"] = df_wien["V"]/(df_wien["mA"] * 1e-3)
df_wien["alt_temp"] = det_temp(df_wien["R"])
df_wien.loc[:, ["V", "mA", "Temp", "R", "alt_temp"]]
# %%
### RESISTANCE METHOD
chosen_point = df_wien.loc[2, ["V", "mA", "Temp", "R"]]
R_0 = chosen_point.V
T_0 = chosen_point.Temp
df_wien["alt_temp"] = det_temp(df_wien["R"])
df_wien

# %%
### EXAMPLE MANUAL FIT
v1 = 8 * np.pi * plank_const * light_speed
v2 = np.divide(plank_const * light_speed, boltzmann_const)
def new_wave_func(x, t):
    val1 = np.divide(v1 , np.power(x, 5))
    val2 = np.exp(np.divide(v2,x * t)) -1 
    return np.divide(val1, val2)


wavelengths = np.linspace(30*1e-9, 1000*1e-9, 1000)
T = 4500
plt.figure(figsize = (10, 8))
intensity_vals = [new_wave_func(i, T)/80 for i in wavelengths]
plt.plot(wavelengths, intensity_vals)
print(wavelengths[intensity_vals.index(max(intensity_vals))], max(intensity_vals))
for i in range(7):
    temp = df_lights[i]
    plt.plot(temp["lambda"], temp["Count"], label = f"{df_pow.V[i]} V, {df_pow.mA[i]} mA")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Count")
plt.title("Blackbody Spectrum of an LED")
plt.legend()
plt.grid()
plt.savefig("exp2_1.png")
# %%
### WAVEFUNCTION?
def wave_function(x, t):
    v1 = 8 * np.pi * plank_const * light_speed
    v2 = np.divide(plank_const * light_speed, boltzmann_const)
    val1 = np.divide(v1 , np.power(x, 5))
    val2 = np.exp(np.divide(v2,x * t)) -1 
    return np.divide(val1, val2)
temp = df_lights[4]
def diff(x):
    t = x[0]
    a = x[1]
    temp = df_lights[1]
    total = 0
    for l,c in zip(temp["lambda"], temp["Count"]):
        total += (a * wave_function(l, t) - c) ** 2
    return total
pot = sc.minimize(diff,np.array([1000, 0.00001]), method='Nelder-Mead', bounds=((100, 8000),(0.000000001, 0.0000001)))
pot
# %%
def give_func(x):
    a, b = pot.x
    counts = []
    for l in temp["lambda"]:
        counts.append(a * wave_function(l, t))
    return counts
plt.figure(figsize=(10, 8))
plt.plot(temp["lambda"], give_func(temp["lambda"]))
plt.plot(temp["lambda"], temp["Count"])
plt.yscale("log")
plt.title("Difference between best fit")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Count")
plt.grid()
plt.savefig("exp2_2.png")
# %%
first = np.divide(2 * plank_const, light_speed**2)
# %%
np.divide(first, 1.71497435e-28)
# %%

# %%
plt.rcParams.update({'font.size': 14})
# %%
plt.figure(figsize = (10, 8))
plt.scatter(df_wien["V"][1:], df_wien["mA"][1:])
plt.plot(df_wien["V"][1:], df_wien["mA"][1:], color = "orange", linestyle = "--")
plt.ylabel("Current (mA)")
plt.xlabel("Voltage (V)")
plt.title("Current over Voltage")
plt.xticks(range(3, 14))
plt.legend()
plt.grid()
plt.savefig("exp1_4.png")
# %%
