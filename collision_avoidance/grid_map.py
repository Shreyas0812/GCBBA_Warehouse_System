import numpy as np
import yaml
import os

class GridMap:
    """
    Warehouse config to grid-based occupancy map
    """

    def __init__(self, config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)

        # Access the parameters
        params = config['create_gridworld_node']['ros__parameters']

        self.width = int(params['grid_width'])
        self.height = int(params['grid_height'])
        self.depth = int(params['grid_depth'])
        self.resolution = float(params['grid_resolution'])

        # Create occupancy grid (0 = free, 1 = obstacle, 2 = induct station, 3 = eject station)
        self.grid = np.zeros((self.depth, self.height, self.width), dtype=np.uint8)

        # Mark obstacles in the grid
        obstacles_regions_flat = params['obstacle_regions']
        self.mark_obstacles(obstacles_regions_flat)
        
        # Mark induct and eject stations
        induct_stations_flat = params['induct_stations']
        self.mark_induct_stations(induct_stations_flat)
        
        eject_stations_flat = params['eject_stations']
        self.mark_eject_stations(eject_stations_flat)

        total_cells = self.width * self.height * self.depth
        obstacle_cells = np.sum(self.grid == 1)
        induct_cells = np.sum(self.grid == 2)
        eject_cells = np.sum(self.grid == 3)
        print("Grid Initialized:")
        print(f" - Dimensions: {self.width} x {self.height} x {self.depth}")
        print(f" - Total Cells: {total_cells}")
        print(f" - Obstacle Cells: {obstacle_cells}")
        print(f" - Induct Stations: {induct_cells}")
        print(f" - Eject Stations: {eject_cells}")
        print(f" - Resolution: {self.resolution}")


    def mark_obstacles(self, obstacles_regions_flat):
        """
        Marks obstacle regions in the occupancy grid.
        
        :param obstacles_regions_flat: [start_x, start_y, start_z, end_x, end_y, end_z, ...]
        """
        num_regions = len(obstacles_regions_flat) // 6
        for i in range(num_regions):
            x_min = int(obstacles_regions_flat[i * 6 + 0] / self.resolution)
            y_min = int(obstacles_regions_flat[i * 6 + 1] / self.resolution)
            z_min = int(obstacles_regions_flat[i * 6 + 2] / self.resolution)
            x_max = int(obstacles_regions_flat[i * 6 + 3] / self.resolution)
            y_max = int(obstacles_regions_flat[i * 6 + 4] / self.resolution)
            z_max = int(obstacles_regions_flat[i * 6 + 5] / self.resolution)

            for z in range(z_min, z_max + 1):
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        if self._in_bounds(x, y, z):
                            self.grid[z, y, x] = 1  # Mark as occupied
    
    def mark_induct_stations(self, induct_stations_flat):
        """
        Marks induct (pickup) stations in the occupancy grid.
        
        :param induct_stations_flat: [x, y, z, station_id, ...]
        """
        num_stations = len(induct_stations_flat) // 4
        for i in range(num_stations):
            x = int(induct_stations_flat[i * 4 + 0] / self.resolution)
            y = int(induct_stations_flat[i * 4 + 1] / self.resolution)
            z = int(induct_stations_flat[i * 4 + 2] / self.resolution)
            if self._in_bounds(x, y, z):
                self.grid[z, y, x] = 2  # Mark as induct station
    
    def mark_eject_stations(self, eject_stations_flat):
        """
        Marks eject (dropoff) stations in the occupancy grid.
        
        :param eject_stations_flat: [x, y, z, station_id, ...]
        """
        num_stations = len(eject_stations_flat) // 4
        for i in range(num_stations):
            x = int(eject_stations_flat[i * 4 + 0] / self.resolution)
            y = int(eject_stations_flat[i * 4 + 1] / self.resolution)
            z = int(eject_stations_flat[i * 4 + 2] / self.resolution)
            if self._in_bounds(x, y, z):
                self.grid[z, y, x] = 3  # Mark as eject station

    def _in_bounds(self, x, y, z):
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth
    

    def continuous_to_grid(self, x, y, z):
        """
        Converts continuous coordinates to grid indices.
        
        :param x: Continuous x coordinate
        :param y: Continuous y coordinate
        :param z: Continuous z coordinate
        :return: (grid_x, grid_y, grid_z)
        """
        grid_x = int(round(x / self.resolution))
        grid_y = int(round(y / self.resolution))
        grid_z = int(round(z / self.resolution))
        return (grid_x, grid_y, grid_z)
    
    def grid_to_continuous(self, grid_x, grid_y, grid_z):
        """
        Converts grid indices to continuous coordinates.
        
        :param grid_x: Grid x index
        :param grid_y: Grid y index
        :param grid_z: Grid z index
        :return: (x, y, z)
        """
        x = grid_x * self.resolution
        y = grid_y * self.resolution
        z = grid_z * self.resolution
        return (x, y, z)
    
    def is_valid_cell(self, grid_x, grid_y, grid_z):
        """
        Checks if a grid cell is valid (within bounds and not an obstacle).
        
        :param grid_x: Grid x index
        :param grid_y: Grid y index
        :param grid_z: Grid z index
        :return: True if valid, False otherwise
        """
        if not self._in_bounds(grid_x, grid_y, grid_z):
            return False
        if self.grid[grid_z, grid_y, grid_x] == 1:
            return False
        return True
    
    def is_induct_station(self, grid_x, grid_y, grid_z):
        """
        Checks if a grid cell is an induct station.
        
        :param grid_x: Grid x index
        :param grid_y: Grid y index
        :param grid_z: Grid z index
        :return: True if induct station, False otherwise
        """
        if not self._in_bounds(grid_x, grid_y, grid_z):
            return False
        return self.grid[grid_z, grid_y, grid_x] == 2
    
    def is_eject_station(self, grid_x, grid_y, grid_z):
        """
        Checks if a grid cell is an eject station.
        
        :param grid_x: Grid x index
        :param grid_y: Grid y index
        :param grid_z: Grid z index
        :return: True if eject station, False otherwise
        """
        if not self._in_bounds(grid_x, grid_y, grid_z):
            return False
        return self.grid[grid_z, grid_y, grid_x] == 3

    def get_neighbors(self, grid_x, grid_y, grid_z):
        """
        Gets valid neighboring cells (6-connectivity).
        
        :param grid_x: Grid x index
        :param grid_y: Grid y index
        :param grid_z: Grid z index
        :return: List of valid neighbor coordinates [(x1, y1, z1), (x2, y2, z2), ...]
        """
        neighbors = []
        directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        
        current_cell_type = self.grid[grid_z, grid_y, grid_x]

        for dx, dy, dz in directions:
            nx, ny, nz = grid_x + dx, grid_y + dy, grid_z + dz
            if not self.is_valid_cell(nx, ny, nz):
                continue
                
            neighbor_cell_type = self.grid[nz, ny, nx]    
            
            if current_cell_type == 2 and neighbor_cell_type == 2:
                continue
            if current_cell_type == 3 and neighbor_cell_type == 3:
                continue

            neighbors.append((nx, ny, nz))
        
        return neighbors

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config', 'gridworld_warehouse_small.yaml')
    grid_map = GridMap(config_path)
    print("Grid map created with dimensions:", grid_map.width, "x", grid_map.height, "x", grid_map.depth)