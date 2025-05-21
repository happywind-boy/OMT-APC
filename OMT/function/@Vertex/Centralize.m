function [V, Center] = Centralize(V)
Center = mean(V, 1);
V = bsxfun(@minus, V, Center);