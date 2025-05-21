function Weight = Gray2Weight(Gray, Scale)
if nargin == 1
    Scale = 1;
end
Weight = Gray / max(Gray);
Weight = exp(Scale*Weight);
Weight = double(Weight);