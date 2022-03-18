function result = fact2(n)
    if (n == 1 || n==0)
        result = 1;
    else
        if (mod(n,2) == 0) % the number is even
            result = prod(2:2:n);
        else % the number is odd
            result = prod(1:2:n);
        end
    end
end