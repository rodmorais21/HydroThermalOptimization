import numpy as np
import time

# Initialization of parameters for robustness study
nr_corridas = 1
media = 0
corridas_best = np.zeros(nr_corridas)
tempos = np.zeros(nr_corridas)

# Parameters of the algorithm
NP = 6    # Number of periods
POP = 20  # Population size
tau_pesos = 0.0005
n = 0
n_ite = 5000

Af = np.array([35, 60, 20, 10, 50, 10])  # Inflows (x10^3)
Load = np.array([70, 80, 130, 50, 70, 110])  # Loads (MW)
CapMax = 150  # Maximum reservoir capacity (x10^3)
VHmax = 80  # Maximum turbine volume per period (x10^3)
Vinicial = 40  # Initial reservoir volume

Ca = 2000  # Cost coefficients for thermal generation
Cb = 100
Cc = 1.5

PTmin = 0  # Minimum thermal generation limit (MW)
PTmax = 80  # Maximum thermal generation limit (MW)

K_turb = 1e10
K_ptmin = 1e10
K_ptmax = 1e10
const_media_termica = 340

melhores = np.zeros((NP + 4, n_ite))
w4_ger = np.zeros(n_ite)

# Initialization of the swarm
v = np.zeros((NP + 4, POP))
vant = np.zeros((NP + 4, POP))
best = np.zeros((NP + 4, POP))
filho_corrente = np.zeros((NP + 4, POP))
filho_clone = np.zeros((NP + 4, POP))

# Fill current particles
for i in range(POP):
    v[:NP, i] = np.random.rand(NP)

# Fill weights
for i in range(POP):
    v[NP:NP+3, i] = 0.05  # Initial weight values

w4 = 0.05

# Previous particles are equal to current
vant[:] = v

# Calculate costs of current particles
for i in range(POP):
    VolSobr = Vinicial
    Custo = 0
    for j in range(NP):
        Voldisp = VolSobr + Af[j]
        Voldisp = min(Voldisp, CapMax)
        VolTurb = v[j, i] * Voldisp
        VolSobr = Voldisp - VolTurb
        PH = (10 / 3) * (VolTurb * 1e3 / 3600)
        PT = Load[j] - PH

        # Constraints
        if VolTurb > VHmax:
            Custo += K_turb * (VolTurb - VHmax) ** 2
        if PT < PTmin:
            Custo += K_ptmin * (PT ** 2)
        if PT > PTmax:
            Custo += K_ptmax * (PTmax - PT) ** 2
        
        Custo += (Ca + Cb * PT + Cc * PT ** 2)
    Psobr = (10 / 3) * (VolSobr * 1e3 / 3600)
    v[NP + 3, i] = Custo - (Psobr * const_media_termica)  # Corrected index

# Best particle in the swarm
bg = np.random.randint(POP)

start_time = time.time()
while n < n_ite:
    n += 1

    # Replication
    v_clone = np.copy(v)

    # Mutation
    for i in range(POP):
        v_clone[NP:NP+3, i] += tau_pesos * np.random.normal(0, 1, 3)
    
    w4 += tau_pesos * np.random.normal(0, 1)

    # Mutation on the best
    best[:NP, bg] += w4 * np.random.normal(0, 1, NP)

    # Reproduction
    for j in range(POP):
        for i in range(NP):
            Nova = (v_clone[NP, j] * (v[i, j] - vant[i, j]) +
                    v_clone[NP+1, j] * (best[i, j] - v[i, j]) +
                    v_clone[NP+2, j] * (best[i, bg] - v[i, j]))
            filho_clone[i, j] = v[i, j] + Nova
            filho_clone[i, j] = np.clip(filho_clone[i, j], 0, 1)
        
        filho_clone[NP:NP+3, j] = v_clone[NP:NP+3, j]

    for j in range(POP):
        for i in range(NP):
            Nova = (v[NP, j] * (v[i, j] - vant[i, j]) +
                    v[NP+1, j] * (best[i, j] - v[i, j]) +
                    v[NP+2, j] * (best[i, bg] - v[i, j]))
            filho_corrente[i, j] = v[i, j] + Nova
            filho_corrente[i, j] = np.clip(filho_corrente[i, j], 0, 1)
        
        filho_corrente[NP:NP+3, j] = v[NP:NP+3, j]

    # Evaluation of offspring
    for i in range(POP):
        for offspring in [filho_corrente, filho_clone]:
            VolSobr = Vinicial
            Custo = 0
            for j in range(NP):
                Voldisp = VolSobr + Af[j]
                Voldisp = min(Voldisp, CapMax)
                VolTurb = offspring[j, i] * Voldisp
                VolSobr = Voldisp - VolTurb
                PH = (10 / 3) * (VolTurb * 1e3 / 3600)
                PT = Load[j] - PH

                # Constraints
                if VolTurb > VHmax:
                    Custo += K_turb * (VolTurb - VHmax) ** 2
                if PT < PTmin:
                    Custo += K_ptmin * (PT ** 2)
                if PT > PTmax:
                    Custo += K_ptmax * (PTmax - PT) ** 2
                
                Custo += (Ca + Cb * PT + Cc * PT ** 2)
            Psobr = (10 / 3) * (VolSobr * 1e3 / 3600)
            offspring[NP + 3, i] = Custo - (Psobr * const_media_termica)

    # Selection
    for k in range(POP):
        if filho_clone[NP + 3, k] < filho_corrente[NP + 3, k]:
            filho_corrente[:, k] = filho_clone[:, k]
        if filho_corrente[NP + 3, k] < v[NP + 3, k]:
            best[:, k] = filho_corrente[:, k]
        else:
            best[:, k] = v[:, k]
    
    # Update global best
    cmin = np.min(best[NP + 3])
    bg = np.argmin(best[NP + 3])

    vant = np.copy(v)
    v = np.copy(best)

    melhores[:, n-1] = best[:, bg]
    w4_ger[n-1] = w4

time_elapsed = time.time() - start_time
tempos[corrida] = time_elapsed
corridas_best[corrida] = best[NP + 3, bg]
media += best[NP + 3, bg]

# Print results
print("Optimal turbine volumes for six periods (in percentages of available water):")
print(best[:NP, bg])

print("\nMinimum total thermal generation cost:")
print(best[NP + 3, bg])

VolSobr = Vinicial
for j in range(NP):
    Voldisp = VolSobr + Af[j]
    Voldisp = min(Voldisp, CapMax)
    VolTurb = best[j, bg] * Voldisp
    VolSobr = Voldisp - VolTurb
print("\nRemaining water in the reservoir after the last period (in m³):")
print(VolSobr * 1000)

# Print the power produced by hydropower in each period
print("\nHydropower production in each period (in MW):")
hydropower = []
VolSobr = Vinicial
for j in range(NP):
    Voldisp = VolSobr + Af[j]
    Voldisp = min(Voldisp, CapMax)
    VolTurb = best[j, bg] * Voldisp
    VolSobr = Voldisp - VolTurb
    PH = (10 / 3) * (VolTurb * 1e3 / 3600)
    hydropower.append(PH)
print(hydropower)

# Print the thermal power produced in each period
print("\nThermal power production in each period (in MW):")
thermal_power = [Load[j] - hydropower[j] for j in range(NP)]
print(thermal_power)

# Print the total thermal generation cost across all periods
total_cost = best[NP + 3, bg]
print("\nTotal thermal generation cost across all periods (€):")
print(total_cost)
