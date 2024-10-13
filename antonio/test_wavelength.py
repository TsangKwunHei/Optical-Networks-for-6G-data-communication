# pyright: reportImplicitRelativeImport=false
from main import find_available_wavelengths, merge_wavelenghts

occupied = [(1, 20), (30, 35)]
available = find_available_wavelengths(occupied, 1)
print(available)
occupied = [(1, 23), (35, 40)]
available = merge_wavelenghts(available, find_available_wavelengths(occupied, 6), 6)
print(available)
