function B = UnfoldTensor3(A, index)

[l, m, n] = size(A);

switch index
    case 1
        B = zeros(l, m * n);
        for i = 1 : m
            B(:, (i - 1) * n + 1 : i * n) = reshape(A(:, i, :), l, n);
        end
    case 2
        B = zeros(m, l * n);
        for i = 1 : n
            B(:, (i - 1) * l + 1 : i * l) = reshape(A(:, :, i), l, m)';
        end
    case 3
        B = zeros(n, l * m);
        for i = 1 : l
            B(:, (i - 1) * m + 1 : i * m) = reshape(A(i, :, :), m, n)';
        end
end

end