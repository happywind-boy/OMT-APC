function [S1, S2, S3, U1, U2, U3] = HOSVD(A, ev)
[U1, S1, ~] = svds(UnfoldTensor3(A, 1), ev);
[U2, S2, ~] = svds(UnfoldTensor3(A, 2), ev);
[U3, S3, ~] = svds(UnfoldTensor3(A, 3), ev);

U1(:, U1(1, :) < 0) = -U1(:, U1(1, :) < 0);
U2(:, U2(1, :) < 0) = -U2(:, U2(1, :) < 0);
U3(:, U3(1, :) < 0) = -U3(:, U3(1, :) < 0);
end
