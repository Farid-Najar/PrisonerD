function sum(a)
    s = 0.
    for x in a
        s += x
    end
    return s
end

print(sum([1,2]))

#=
Multi line comment
=#
s = 10
print("Far is $s years old")
print("Far is $(2*s + 4) years old")

# NOTE : string concatenation operator in julia is *
print("macron"*" demission")

named_tuple = (name = "farid", age = 24)

print(named_tuple[1])
print(named_tuple.name)

x = fill(0, (2,3))
y = ones(5)

print(y)
(x->x*2).(y)
f = x->x*2
broadcast!(f, y, y)

using Pkg
Pkg.add("PyCall")

g(x) = x.^2
g([2])