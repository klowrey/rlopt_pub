
n = 20
T = 1000
K = 100

#x = zeros(K, T, n)
x = zeros(n, T, K)
y = randn(n)


function fill(x, y)
   #K = size(x, 1)
   #T = size(x, 2)
   T = size(x, 2)
   K = size(x, 3)

   for k=1:K
      for t=1:T
         x[:,t,k] = y
         #x[k,t,:] = y
      end
   end
end

@time fill(x,y)
Profile.clear_malloc_data()
@time fill(x,y)
