__precompile__()

module ExpFunctions

struct FunctionSet
   setmodel!::Function
   initstate!::Function
   setcontrol!::Function
   observe!::Function
   reward::Function
   evaluate::Function
end

export FunctionSet


end

