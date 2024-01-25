import numpy as np
import matplotlib.pyplot as plt

class DecisionBoundariesPlotter:
	def __init__(self, model, name_x, name_y, variables_to_use, projection_values_variables_to_use, minimum_variables, maximum_variables, variables, path):
		self.model = model
		self.name_x = name_x
		self.name_y = name_y
		self.variables = variables
		self.variables_to_use = variables_to_use
		self.projection_values_variables_to_use = projection_values_variables_to_use
		self.minimum_variables = minimum_variables
		self.maximum_variables = maximum_variables
		self.path = path
		self.indices_found = None
		self.shape = 200

	def find_variables(self):
		indices = [i for i, var in enumerate(self.variables_to_use) if var == self.name_x or var == self.name_y]
		found_variables = [{"name": var, "min": self.minimum_variables[i], "max": self.maximum_variables[i]} for i, var in enumerate(self.variables_to_use) if var in [self.name_x, self.name_y]]
		not_found_variables = [var for var in self.variables_to_use if var not in [self.name_x, self.name_y]]

		self.indices_found = indices
		xy = {"indices": indices, "variables": found_variables}
		projection = {"indices": indices, "variables": found_variables, "projection_values": [float(val) for val in self.projection_values_variables_to_use[indices[0]:indices[0]+len(indices)]]}

	def create_grid(self):
		if self.indices_found is None:
			raise ValueError("Indices not found. Run find_variables method first.")

		var_x_min = float(self.minimum_variables[self.indices_found[0]])
		var_x_max = float(self.maximum_variables[self.indices_found[0]])
		var_y_min = float(self.minimum_variables[self.indices_found[1]])
		var_y_max = float(self.maximum_variables[self.indices_found[1]])

		c_grid, e_grid = np.meshgrid(np.linspace(var_x_min, var_x_max, self.shape), np.linspace(var_y_min, var_y_max, self.shape))
		
		return c_grid, e_grid

	def generate_input_arrays(self, c_grid, e_grid):
		counter = 0
		if self.variables == 7:
			r1, r2, r3, r4, r5, r6, r7 = (
			np.array([]), np.array([]), np.array([]),
			np.array([]), np.array([]), np.array([]),
			np.array([])  # Inizializza r7 come array vuoto
			)
			r_arrays = [r1, r2, r3, r4, r5, r6, r7]

		elif self.variables == 5:
			r1, r2, r3, r4, r5 = (
			np.array([]), np.array([]), np.array([]),
			np.array([]), np.array([])  # Inizializza r7 come array vuoto
			)
			r_arrays = [r1, r2, r3, r4, r5]

		
		size = len(c_grid.flatten())
		

		for i, array in enumerate(r_arrays):
			index_to_check = i
		
			if index_to_check in self.indices_found:
				if counter == 0: 
					r_arrays[i] = c_grid.flatten()
					r_arrays[i] = r_arrays[i].reshape(len(r_arrays[i]),1)
					counter = counter + 1
				elif counter == 1: 
					r_arrays[i] = e_grid.flatten()
					r_arrays[i] = r_arrays[i].reshape(len(r_arrays[i]),1)		
		
			else:
				r_arrays[i] = np.full((size,1), fill_value=self.projection_values_variables_to_use[i]) 

		if self.variables == 5:
			grid = np.hstack((r_arrays[0], r_arrays[1], r_arrays[2], r_arrays[3], r_arrays[4]))
		elif self.variables == 7:
			grid = np.hstack((r_arrays[0], r_arrays[1], r_arrays[2], r_arrays[3], r_arrays[4], r_arrays[5], r_arrays[6]))
		else:
			raise ValueError("Not acceptable number of variables")


		return grid

	def predict_and_plot(self, grid):
		prediction = self.model.predict(grid)

		prediction_clabel = prediction[:, 0].reshape((self.shape, self.shape))

		fig, ax = plt.subplots()
		CS = plt.contour(self.c_grid, self.e_grid, prediction_clabel, colors='black')
		ax.clabel(CS, CS.levels, inline=True, fontsize=10)
		var_x_min = float(self.minimum_variables[self.indices_found[0]])
		var_x_max = float(self.maximum_variables[self.indices_found[0]])
		var_y_min = float(self.minimum_variables[self.indices_found[1]])
		var_y_max = float(self.maximum_variables[self.indices_found[1]])
		
		plt.imshow(prediction_clabel, extent=(var_x_min, var_x_max, var_y_min, var_y_max),
		   cmap='bwr_r', origin='lower', aspect='auto', alpha=0.5)
		#plt.hlines(1.5, var_x_min, var_x_max, linestyle='dashed', color='black',
			   #label='Canonical IBD cut')

		#plt.imshow(prediction_clabel, extent=(self.var_x_min, self.var_x_max, self.var_y_min, self.var_y_max),
		#   cmap='bwr_r', origin='lower', aspect='auto', alpha=0.5)
		#plt.hlines(self.var_y_max, self.var_x_min, self.var_x_max, linestyle='dashed', color='lightblue',
		#	   label='Canonical IBD cut')

		ax.set_xlabel(self.variables_to_use[self.indices_found[0]])
		ax.set_ylabel(self.variables_to_use[self.indices_found[1]])
		ax.set_title(f'2D decision boundaries ($r^3$ = 1000 m$^3$, $E_d$ = 2.2 MeV, m_QL = 20000)')

		rectangle = plt.Rectangle((var_x_min, var_y_min), var_x_max - var_x_min, var_y_max - var_y_min, fill=None, edgecolor='black', linewidth=2, linestyle='dashed', label="Canonical IBD cut")
		ax.add_patch(rectangle)

		ax.grid()
		plt.legend(loc='best')
		plt.savefig(self.path + "_DecisionBoundaries_" + self.variables_to_use[self.indices_found[0]] + "_" + self.variables_to_use[self.indices_found[1]] + ".pdf")
		plt.close()

		#ax.set_ylabel(f'$\Delta r$ [m]')
		#ax.set_xlabel(f'$E_p$ [MeV]')
		#ax.set_title(f'Neural network decision boundaries ($r^3$ = 1000 m$^3$, $E_d$ = 2.2 MeV, m_QL = 20000)')
		#ax.grid()
		#plt.legend(loc='best')
		#plt.savefig(self.path + "_DecisionBoundaries_Ep_DeltaR.pdf")
		#plt.close()



	def run(self):
		self.find_variables()
		self.c_grid, self.e_grid = self.create_grid()
		grid = self.generate_input_arrays(self.c_grid, self.e_grid)
		self.predict_and_plot(grid)


