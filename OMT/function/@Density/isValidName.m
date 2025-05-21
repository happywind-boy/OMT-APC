function Valid = isValidName(DensityName)
    Valid = any(strcmpi(DensityName, Density.DensityType));
end