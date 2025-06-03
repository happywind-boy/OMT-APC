% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function Sigma = StretchFactor(T, V, U)
Vol_V = abs(Tet.Volume(T, V)); % abs is necessary
Vol_U = abs(Tet.Volume(T, U)); % abs is necessary
Vol_V = Vol_V / sum(Vol_V);
Vol_U = Vol_U / sum(Vol_U);
Sigma = Vol_V ./ Vol_U;