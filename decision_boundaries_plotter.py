import numpy as np
import matplotlib.pyplot as plt

class DecisionBoundariesPlotter:
    def __init__(self, model, Ep_Threshold_Min, Ep_Threshold_Max, variables, DeltaR_Max, path):
        self.model = model
        self.Ep_Threshold_Min = Ep_Threshold_Min
        self.Ep_Threshold_Max = Ep_Threshold_Max
        self.variables = variables
        self.DeltaR_Max = DeltaR_Max
        self.path = path

    def create_grid(self):
        c_grid, e_grid = np.meshgrid(np.linspace(self.Ep_Threshold_Min, self.Ep_Threshold_Max, 10),
                                     np.linspace(0, 4, 10))
        return c_grid, e_grid

    def generate_input_arrays(self, c_grid, e_grid):
        r3, r5 = c_grid.flatten(), e_grid.flatten()
        r3, r5 = r3.reshape((len(r3), 1)), r5.reshape((len(r5), 1))
        r1 = np.full((len(r3), 1), 1000)
        r2 = np.full((len(r3), 1), 1000)
        r4 = np.full((len(r3), 1), 2.2)
        r6 = np.full((len(r3), 1), 20000)
        r7 = np.full((len(r3), 1), 20000)

        if self.variables == 5:
            grid = np.hstack((r1, r2, r3, r4, r5))
        elif self.variables == 7:
            grid = np.hstack((r1, r2, r3, r4, r5, r6, r7))
        else:
            raise ValueError("Not acceptable number of variables")

        return grid

    def predict_and_plot(self, grid):
        prediction = self.model.predict(grid)
        prediction_clabel = prediction[:, 0].reshape((10, 10))

        fig, ax = plt.subplots()
        CS = plt.contour(self.c_grid, self.e_grid, prediction_clabel, colors='black')
        ax.clabel(CS, CS.levels, inline=True, fontsize=10)
        plt.imshow(prediction_clabel, extent=(self.Ep_Threshold_Min, self.Ep_Threshold_Max, 0., 4),
                   cmap='bwr_r', origin='lower', aspect='auto', alpha=0.5)
        plt.hlines(self.DeltaR_Max, self.Ep_Threshold_Min, self.Ep_Threshold_Max, linestyle='dashed', color='lightblue',
                   label='Canonical IBD cut')

        ax.set_ylabel(f'$\Delta r$ [m]')
        ax.set_xlabel(f'$E_p$ [MeV]')
        ax.set_title(f'Neural network decision boundaries ($r^3$ = 1000 m$^3$, $E_d$ = 2.2 MeV, m_QL = 20000)')
        ax.grid()
        plt.legend(loc='best')
        plt.savefig(self.path + "_DecisionBoundaries_Ep_DeltaR.pdf")
        plt.close()

    def plot_grids(self):
        print(self.c_grid)
        print(self.e_grid)

    def run(self):
        self.c_grid, self.e_grid = self.create_grid()
        grid = self.generate_input_arrays(self.c_grid, self.e_grid)
        self.predict_and_plot(grid)
        self.plot_grids()


# Esempio di utilizzo:
# Sostituisci 'model', 'Ep_Threshold_Min', 'Ep_Threshold_Max', 'variables', 'DeltaR_Max', 'path'
# con i tuoi valori effettivi.
# decision_boundaries_plotter = DecisionBoundariesPlotter(model, Ep_Threshold_Min, Ep_Threshold_Max, variables, DeltaR_Max, path)
# decision_boundaries_plotter.run()
