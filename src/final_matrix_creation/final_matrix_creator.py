from data_structures.compound_pixel_grid import CompoundExpandablePixelGrid
from data_structures.vectors import Position2D
import copy
import math
import skimage

import numpy as np
import cv2 as cv

from flags import SHOW_MAP_AT_END, DO_SAVE_FINAL_MAP, SAVE_FINAL_MAP_DIR, DO_SAVE_DEBUG_GRID, SAVE_DEBUG_GRID_DIR
from mapping import mapper
import time
#from fixture_detection.fixture_clasification import FixtureClasiffier
#from fixture_detection.fixture_detection import FixtureDetector

class WallMatrixCreator:
    def __init__(self, square_size_px: int):
        self.threshold = 10
        self.__square_size_px = square_size_px

        straight = [
            [0, 0, 1, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 1, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
        
        self.straight_template = np.array(straight)

        
        vortex = [
            [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
        
        self.vortex_template = np.array(vortex)
        


        self.templates = {}

        for i, name in enumerate([(-1, 0), (0,-1), (1,0), (0,1)]):
            self.templates[name] = np.rot90(self.straight_template, i)
        
        for i, name in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):
           self.templates[name] = np.rot90(self.vortex_template, i)
        

    def __get_tile_status(self, min_x, min_y, max_x, max_y, wall_array: np.ndarray) -> list:
        counts = {name: 0 for name in self.templates}
        square = wall_array[min_x:max_x, min_y:max_y]
        if square.shape != (self.__square_size_px, self.__square_size_px):
            return []

        non_zero_indices = np.where(square != 0)
        for orientation, template in self.templates.items():
            counts[orientation] = np.sum(template[non_zero_indices])

        status = []
        for orientation, count in counts.items():
            if count >= self.threshold:
                status.append(orientation)

        return status

    def transform_wall_array_to_bool_node_array(self, wall_array: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        grid = []
        if SHOW_MAP_AT_END:
            bool_array_copy = wall_array.astype(np.uint8) * 100
        for x in range(offsets[0], wall_array.shape[0] - self.__square_size_px, self.__square_size_px):
            row = []
            for y in range(offsets[1], wall_array.shape[1] - self.__square_size_px, self.__square_size_px):
                min_x = x
                min_y = y
                max_x = x + self.__square_size_px
                max_y = y + self.__square_size_px
                #print(min_x, min_y, max_x, max_y)
                if SHOW_MAP_AT_END:
                    bool_array_copy = cv.rectangle(bool_array_copy, (min_y, min_x), (max_y, max_x), (255,), 1)
                
                val = self.__get_tile_status(min_x, min_y, max_x, max_y, wall_array)
                
                row.append(list(val))
            grid.append(row)
        
        if SHOW_MAP_AT_END:
            cv.imshow("point_cloud_with_squares", cv.resize(bool_array_copy, (0, 0), fx=1, fy=1, interpolation=cv.INTER_AREA))

        grid = self.__orientation_grid_to_final_wall_grid(grid)

        return grid
    
    def __orientation_grid_to_final_wall_grid(self, orientation_grid: list) -> np.ndarray:
        shape = np.array([len(orientation_grid), len(orientation_grid[0])])
        shape *= 2

        final_wall_grid = np.zeros(shape, dtype=np.bool_)
        
        for y, row in enumerate(orientation_grid):
            for x, value in enumerate(row):
                x1 = x * 2
                y1 = y * 2

                for orientation in value:
                    final_x = x1 + orientation[1]
                    final_y = y1 + orientation[0]

                    final_wall_grid[final_y, final_x] = True
        
        return final_wall_grid
    
#    def victim_in_grid(self):
#        letter_position = FixtureDetector.get_fixture_positions_in_image("Y") * "ancho_de_matriz" + FixtureDetector.get_fixture_positions_in_image("X")
#        return letter_position



class FloorMatrixCreator:
    def __init__(self, square_size_px: int) -> None:
        self.__square_size_px = square_size_px * 2
        self.__floor_color_ranges = {
                    "0": # Normal
                        {   
                            "range":   ((0, 0, 37), (0, 0, 192)), 
                            "threshold":0.2},

                    "0": # Void
                        {
                            "range":((100, 0, 0), (101, 1, 1)),
                            "threshold":0.9},
                    
                    "4": # Checkpoint
                        {
                            "range":((95, 0, 65), (128, 122, 198)),
                            "threshold":0.2},
                    "2": # Hole
                        {
                            "range":((0, 0, 10), (0, 0, 30)),
                            "threshold":0.2},
                    
                    "3": # swamp
                        {
                            "range":((19, 112, 32), (19, 141, 166)),
                            "threshold":0.2},

                    "6": # Connection 1-2
                        {
                            "range":((120, 182, 49), (120, 204, 232)),
                            "threshold":0.2},

                    "8": # connection 3-4
                        {
                            "range":((132, 156, 36), (133, 192, 185)),
                            "threshold":0.2},

                    "7": # Connection2-3
                        {
                            "range":((0, 182, 49), (0, 204, 232)),
                            "threshold":0.2},
                    }
        
                    #TODO Add Connection 1-4
                    
        self.final_image = np.zeros((700, 700, 3), np.uint8)
        
    def __get_square_color(self, min_x, min_y, max_x, max_y, floor_array: np.ndarray) -> str:
        square = floor_array[min_x:max_x, min_y:max_y]

        square = cv.cvtColor(square, cv.COLOR_BGR2HSV)
        
        if np.count_nonzero(square) == 0:
            return "0"
        
        color_counts = {}
        for color_key, color_range in self.__floor_color_ranges.items():
            colour_count = np.count_nonzero(cv.inRange(square, color_range["range"][0], color_range["range"][1]))
            if colour_count > color_range["threshold"] * square.shape[0] * square.shape[1]:
                color_counts[color_key] = colour_count
        
        if len(color_counts) == 0:
            return "0"
        else:
            return max(color_counts, key=color_counts.get)
    

    def get_floor_colors(self, floor_array: np.ndarray, offsets: np.ndarray) -> np.ndarray:

        if SHOW_MAP_AT_END:
            array_copy = copy.deepcopy(floor_array)

        grid = []

        for x in range(offsets[0], floor_array.shape[0] - self.__square_size_px, self.__square_size_px):
            row = []
            for y in range(offsets[1], floor_array.shape[1] - self.__square_size_px, self.__square_size_px):
                min_x = x
                min_y = y
                max_x = x + self.__square_size_px
                max_y = y + self.__square_size_px
                
                if SHOW_MAP_AT_END:
                    array_copy = cv.rectangle(array_copy, (min_y, min_x), (max_y, max_x), (255, 255, 255), 1)
                
                color_key = self.__get_square_color(min_x, min_y, max_x, max_y, floor_array)

                row.append(color_key)
            grid.append(row)

        if SHOW_MAP_AT_END:
            cv.imshow("array copy", array_copy)

        return grid
        

class FinalMatrixCreator:
    def __init__(self, tile_size: float, resolution: float):
        self.__square_size_px = round(tile_size / 2 * resolution)

        self.wall_matrix_creator = WallMatrixCreator(self.__square_size_px)
        self.floor_matrix_creator = FloorMatrixCreator(self.__square_size_px)

    def stringMatriz(self, matriz):
        #matriz de string a int
        intmatriz = []

        for fila in matriz:
            aux = []
            for elemento in fila:
                a = int(elemento)
                aux.append(a)
            intmatriz.append(aux)

        for i in intmatriz:
            print(i)
        
        return intmatriz
    #def diferencia_mayor_a_10(coordenada1, coordenada2):
        #return abs(coordenada1[0] - coordenada2[0]) > 10 or abs(coordenada1[1] - coordenada2[1]) > 10

    #def punto_victim(self):
        #lista1 = mapper.Mapper.victim_position()
        #lista2 = []
        #indiceuno = 0
        #indicedos = 0
        #for i in range (len(lista1)):
            #for i in range (len(lista1)-1):
                #if self.diferencia_mayor_a_10(lista1[indiceuno], lista1[indicedos]) == True:
                    #lista2.append(lista1[indiceuno])
                    #indicedos += 1
        #indiceuno += 1

    def stringMatrizreverse(self, matriz):
        #matriz de int a str
            stringmatriz = []

            for fila in matriz:
                aux = []
                for elemento in fila:
                    a = str(elemento)
                    aux.append(a)
                stringmatriz.append(aux)

            for i in stringmatriz:
                print(i)
            
            return stringmatriz

    def delete_row(self, matriz_procesar):
    #Quita las filas que contengan valores innecesarios
        columnastotales = len(matriz_procesar[0])
        column_reference = [0]*columnastotales
        
    
        result = [elem for elem in matriz_procesar if elem != column_reference]
        return result
    

    def transposed_matriz2(self, matriz):
        nueva_matriz = [list(columnas) for columnas in zip(*matriz)]
        return nueva_matriz

    def correccion_de_bordes_filas(self, matriz):
        cant_c = (len(matriz[0]))
        columna = 0
        for i in range (0, cant_c):
            lmatrix = matriz[0][columna]
            if lmatrix != "0":
                columna += 1
            elif lmatrix == "0":        
                matriz[0][columna] = "1"
                columna += 1

        columna = 0
        for i in range (0, cant_c):
            lmatrix = matriz[-1][columna]
            if lmatrix != "0":
                columna += 1
            elif lmatrix == "0":        
                matriz[-1][columna] = "1"
                columna += 1
        columna = 0
        return matriz

    def correccion_de_bordes_columnas(self, matriz):
        matriz = [list(columnas) for columnas in zip(*matriz)]
        cant_c = (len(matriz[0]))
        columna = 0
        for i in range (0, cant_c):
            lmatrix = matriz[0][columna]
            if lmatrix != "0":
                columna += 1
            elif lmatrix == "0":        
                matriz[0][columna] = "1"
                columna += 1

        columna = 0
        for i in range (0, cant_c):
            lmatrix = matriz[-1][columna]
            if lmatrix != "0":
                columna += 1
            elif lmatrix == "0":        
                matriz[-1][columna] = "1"
                columna += 1
        columna = 0
        matriz = [list(columnas) for columnas in zip(*matriz)]
        return matriz

    def correccion_de_interioresA(self, matriz):
        #corrige casos en que haya un 0 incorrecto en medio de dos 1
        fila = 0
        columna = 0
        cant_f = (len(matriz))
        cant_c = (len(matriz[0]))
        for i in range (0, cant_f):
            for i in range (0, cant_c):
                lmatrix = matriz[fila][columna]
                if lmatrix != "0":
                    columna += 1
                elif lmatrix == "0":
                    if ((matriz[fila][columna + 1]) == "1") and ((matriz[fila][columna - 1]) == "1"):
                        matriz[fila][columna] = "1"
                        columna += 1

                    elif ((matriz[fila + 1][columna]) == "1") and ((matriz[fila - 1][columna]) == "1"):
                        matriz[fila][columna] = "1"
                        columna += 1
                
                    else:
                        columna+= 1
            columna = 0
            fila += 1
        return matriz

    def correccion_de_interioresB(self, matriz):
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
        
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                
                if lmatrix == "0":
                    if fila == 2 or fila == (cant_f - 2):
                        pass
                    else:
                        try:
                            if ((matriz[fila][columna + 1] == "1") and (matriz[fila][columna - 1] != "1") and (matriz[fila][columna + 2] == "1") and (matriz[fila + 1][columna] == "1") and (matriz[fila + 2][columna] == "1")): 
                                if (((matriz[fila - 1][columna + 4] != "1") and (matriz[fila + 2][columna + 4] != "1")) and \
                                ((matriz[fila + 4][columna - 1] != "1") and (matriz[fila + 4][columna - 2] != "1"))):
                                    
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                        try:
                            if ((matriz[fila][columna - 1] == "1") and (matriz[fila][columna + 1] != "1") and (matriz[fila][columna - 2] == "1") and (matriz[fila + 1][columna] == "1") and (matriz[fila + 2][columna] == "1")):
                                if (((matriz[fila - 1][columna - 4] != "1") and (matriz[fila - 2][columna - 4] != "1")) and \
                                ((matriz[fila + 4][columna + 1] != "1") and (matriz[fila + 4][columna + 2] != "1"))):
                                    
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                columna += 1

            columna = 0
            fila += 1

        return matriz


    def correccion_de_interioresC(self, matriz):
        #corrige vertices interiores hacia abajo
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
        
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                
                if lmatrix == "0":
                    if fila == 1 or fila == (cant_f - 2):
                        pass
                    else:
                        try:
                            if ((matriz[fila][columna + 1] == "1") and (matriz[fila][columna - 1] != "1") and (matriz[fila][columna + 2] == "1") and (matriz[fila - 1][columna] == "1") and (matriz[fila - 2][columna] == "1")):
                                if (((matriz[fila + 1][columna + 4] != "1") and (matriz[fila + 2][columna + 4] != "1")) and \
                                ((matriz[fila - 4][columna - 1] != "1") and (matriz[fila - 4][columna - 2] != "1"))):
                                
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                        try:
                            if ((matriz[fila][columna - 1] == "1") and (matriz[fila][columna + 1] != "1") and (matriz[fila][columna - 2] == "1") and (matriz[fila - 1][columna] == "1") and (matriz[fila - 2][columna] == "1")):
                                if (((matriz[fila + 1][columna-4] != "1") and (matriz[fila + 2][columna - 4] != "1")) and \
                                ((matriz[fila - 4][columna + 1] != "1") and (matriz[fila - 4][columna + 2] != "1"))):
                            
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                columna += 1

            columna = 0
            fila += 1

        return matriz
    
    def correccion_de_interioresD(self, matriz):
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
        
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                if lmatrix == "0":
                    if (fila == 1 or fila == (cant_f - 2)) or (columna == 1 or columna == (columna == (cant_c - 2))):
                        pass
                    else:
                        try:
                            if ((matriz[fila][columna + 1] == "1") and (matriz[fila][columna +2] == "1") and (matriz[fila][columna + 3] != "1")):
                                if ((matriz[fila - 1][columna] != "1") and (matriz[fila + 1][columna] != "1") and (matriz[fila - 1][columna + 3] != "1") and (matriz[fila + 1][columna + 3] != "1")):

                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass
                        try:
                            if ((matriz[fila][columna - 1] == "1") and (matriz[fila][columna - 2] == "1") and (matriz[fila][columna -3] != "1")) :
                                if ((matriz[fila - 1][columna] != "1") and (matriz[fila + 1][columna] != "1") and (matriz[fila - 1][columna - 3] != "1") and (matriz[fila + 1][columna - 3] != "1")):
                            
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                        try:
                            if ((matriz[fila + 1][columna] == "1") and (matriz[fila + 2][columna] == "1") and (matriz[fila + 3][columna ] != "1")):
                                if ((matriz[fila][columna - 1] != "1") and (matriz[fila][columna + 1] != "1") and (matriz[fila + 3][columna - 1] != "1") and (matriz[fila + 3][columna + 1] != "1")):
                                    
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                        try:
                            if ((matriz[fila - 1][columna] == "1") and (matriz[fila - 2][columna] == "1") and (matriz[fila - 3 ][columna] != "1")):
                                if ((matriz[fila][columna - 1] != "1") and (matriz[fila][columna + 1] != "1") and (matriz[fila - 3][columna - 1] != "1") and (matriz[fila - 3][columna + 1] != "1")):
                            
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                columna += 1

            columna = 0
            fila += 1

        return matriz
    def pixel_grid_to_final_grid(self, pixel_grid: CompoundExpandablePixelGrid, robot_start_position: np.ndarray) -> np.ndarray:
        np.set_printoptions(linewidth=1000000000000, threshold=100000000000000)
        wall_array = pixel_grid.arrays["walls"]
        color_array = pixel_grid.arrays["floor_color"]

        if DO_SAVE_FINAL_MAP:
            cv.imwrite(f"{SAVE_FINAL_MAP_DIR}/WALL_PIXEL_GRID{str(time.time()).rjust(50)}.png", wall_array.astype(np.uint8) * 255)

        if DO_SAVE_DEBUG_GRID:
            cv.imwrite(f"{SAVE_DEBUG_GRID_DIR}/DEBUG_GRID{str(time.time()).rjust(50)}.png", (pixel_grid.get_colored_grid() * 255).astype(np.uint8))

        offsets = self.__get_offsets(self.__square_size_px, pixel_grid.offsets)
        
        # Walls
        wall_node_array = self.wall_matrix_creator.transform_wall_array_to_bool_node_array(wall_array, offsets)


        floor_offsets = self.__get_offsets(self.__square_size_px * 2, pixel_grid.offsets + self.__square_size_px)

        # Floor
        floor_string_array = self.floor_matrix_creator.get_floor_colors(color_array, floor_offsets)

        # Start tile
        if robot_start_position is None:
            return np.array([])
        
        start_array_index = pixel_grid.coordinates_to_array_index(robot_start_position)
        start_array_index -= offsets
        robot_node = np.round((start_array_index / self.__square_size_px) * 2).astype(int) - 1


        # Mix everything togehter
        text_grid = self.__get_final_text_grid(wall_node_array, floor_string_array, robot_node)
        text_grid = self.stringMatriz(text_grid)
        text_grid = self.delete_row(text_grid)
        text_grid = self.transposed_matriz2(text_grid)
        text_grid = self.delete_row(text_grid)
        text_grid = self.transposed_matriz2(text_grid)
        text_grid = self.stringMatrizreverse(text_grid)
        text_grid = self.correccion_de_bordes_filas(text_grid)
        text_grid = self.correccion_de_bordes_columnas(text_grid)
        text_grid = self.correccion_de_interioresA(text_grid)
        text_grid = self.correccion_de_interioresB(text_grid)
        text_grid = self.correccion_de_interioresC(text_grid)
        text_grid = self.correccion_de_interioresD(text_grid)

        return np.array(text_grid)
        

        #wall_array = self.offset_array(wall_array, self.square_size_px, pixel_grid.offsets)
        #color_array = self.offset_array(color_array, self.square_size_px, pixel_grid.offsets)

    def __get_final_text_grid(self, wall_node_array: np.ndarray, floor_type_array: np.ndarray, robot_node: np.ndarray) -> list:
        if SHOW_MAP_AT_END:
            cv.imshow("final_grid", cv.resize(wall_node_array.astype(np.uint8) * 255, (0, 0), fx=10, fy=10, interpolation=cv.INTER_AREA))


        if DO_SAVE_FINAL_MAP:
            cv.imwrite(f"{SAVE_FINAL_MAP_DIR}/WALL_GRID{str(time.time()).rjust(50)}.png", wall_node_array.astype(np.uint8) * 255)

        
        final_text_grid = []

        # set walls
        for row in wall_node_array:
            f_row = []
            for val in row:
                if val:
                    f_row.append("1")
                else:
                    f_row.append("0")
            final_text_grid.append(f_row)

#        if WallMatrixCreator.victim_in_grid():
#            if FixtureClasiffier.classify_fixture == "U":
#                f_row.append("U")
#            if FixtureClasiffier.classify_fixture == "S":
#                f_row.append("S")
#            if FixtureClasiffier.classify_fixture == "H":
#                f_row.append("H")
#            final_text_grid.append(f_row)

        #set floor
        for y, row in enumerate(floor_type_array):
            for x, val in enumerate(row):
                x1 = x * 4 + 3
                y1 = y * 4 + 3
                self.__set_node_as_character(final_text_grid, np.array([y1, x1]), val)

        
        self.__set_node_as_character(final_text_grid, robot_node, "5")
        return final_text_grid
        
    
    def __get_offsets(self, square_size: float, raw_offsets: np.array) -> np.ndarray:
        return np.round(raw_offsets % square_size).astype(int)
    

    def __set_node_as_character(self, final_text_grid: list, node: np.ndarray, character: str) -> list:
        for diagonal in np.array(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            n = node + diagonal
            try:
                final_text_grid[n[0]][n[1]] = character
            except IndexError:
                pass

        return final_text_grid

