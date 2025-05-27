import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    # Builds a Möbius strip and calculates its area, edge length, and 3D plot.

    def __init__(self, R, w, n):
        # Sets up the strip with radius R, width w, and n points for smoothness.
        self.R = R  # Radius
        self.w = w  # Width
        self.n = n  # Resolution
        self.u = np.linspace(0, 2 * np.pi, n)  # Angle points (0 to 2pi)
        self.v = np.linspace(-w / 2, w / 2, n)  # Width points
        self.mesh = None  

    def parametric_equations(self, u, v):
        # Turns u and v into 3D (x, y, z) points for the strip.
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)  # x 
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)  # y 
        z = v * np.sin(u / 2)  # z
        return x, y, z  # Returns 3D point

    def generate_mesh(self):
        # Creates a grid of points to shape the Möbius strip.
        U, V = np.meshgrid(self.u, self.v)  # Makes 2D grid
        x, y, z = self.parametric_equations(U, V)  # Gets x, y, z for grid
        self.mesh = (x, y, z)  # Saves grid
        return self.mesh

    def compute_surface_area(self):
        #Compute the surface area 
        if self.mesh is None:
            self.generate_mesh()  

        U, V = np.meshgrid(self.u, self.v)  # Sets up grid

        # Finds slopes to measure each patch’s area.
        dx_du = (-(self.R + V * np.cos(U / 2)) * np.sin(U) - 
                 (V / 2) * np.sin(U / 2) * np.cos(U))  # x slope (u)
        dy_du = ((self.R + V * np.cos(U / 2)) * np.cos(U) - 
                 (V / 2) * np.sin(U / 2) * np.sin(U))  # y slope (u)
        dz_du = (V / 2) * np.cos(U / 2)  # z slope (u)

        dx_dv = np.cos(U / 2) * np.cos(U)  # x slope (v)
        dy_dv = np.cos(U / 2) * np.sin(U)  # y slope (v)
        dz_dv = np.sin(U / 2)  # z slope (v)

        # Uses cross product to get area of each patch.
        cross_prod_x = dy_du * dz_dv - dz_du * dy_dv  # x part
        cross_prod_y = dz_du * dx_dv - dx_du * dz_dv  # y part
        cross_prod_z = dx_du * dy_dv - dy_du * dx_dv  # z part
        integrand = np.sqrt(cross_prod_x**2 + cross_prod_y**2 + cross_prod_z**2) 

        
        du = self.u[1] - self.u[0]  
        dv = self.v[1] - self.v[0]  
        surface_area = np.sum(integrand) * du * dv  # Total area
        return surface_area

    def compute_edge_length(self):
        
        u = self.u  
        v = self.w / 2 
        x, y, z = self.parametric_equations(u, v)  # Edge coordinates

        # Finds how fast the edge curve changes.
        dx_du = np.gradient(x, u)  # x change
        dy_du = np.gradient(y, u)  # y change
        dz_du = np.gradient(z, u)  # z change

        # Calculates length of tiny edge segments.
        integrand = np.sqrt(dx_du**2 + dy_du**2 + dz_du**2) 

        # Adds up segments, scaled by step size.
        du = u[1] - u[0]  
        edge_length = np.sum(integrand) * du  # Total edge length
        return edge_length

    def visualize(self):
        # Draws the Möbius strip in 3D to show its twist.
        if self.mesh is None:
            self.generate_mesh()  # Gets mesh if needed

        x, y, z = self.mesh  # Gets x, y, z points
        fig = plt.figure(figsize=(8, 8))  # Makes a figure
        ax = fig.add_subplot(111, projection='3d')  # Sets up 3D plot
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)  # Draws strip
        ax.set_xlabel('X')  # x-axis label
        ax.set_ylabel('Y')  # y-axis label
        ax.set_zlabel('Z')  # z-axis label
        ax.set_title('Möbius Strip (R={}, w={})'.format(self.R, self.w))  # Title
        plt.show()  # Shows plot


if __name__ == "__main__":
    # Example values to try.
    R = 1.0  # Radius
    w = 0.5  # Width
    n = 100  # Resolution
    
    mobius = MobiusStrip(R, w, n)  # Creates strip
    mobius.generate_mesh()  # Builds 3D grid
    area = mobius.compute_surface_area()  # Gets area
    edge_length = mobius.compute_edge_length()  # Gets edge length
    print(f"Surface Area: {area:.4f}")  # Prints area
    print(f"Edge Length: {edge_length:.4f}")  # Prints edge length
    mobius.visualize()  # Shows 3D plot
