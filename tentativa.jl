function createDataStruct(indexes, data)

  #precisar ser o máximo do total de dados para garantir
  #o tamanho total da matriz
  quantItens = maximum(map(x -> parse(Int32, (split(x, "\t")[2])), data))
  quantUsers = maximum(map(x -> parse(Int32, (split(x, "\t")[1])), data))

  users = map(x -> parse(Int32, (split(data[x], "\t")[1] )), indexes)
  itens = map(x -> parse(Int32, (split(data[x], "\t")[2] )), indexes)
  values = map(x -> parse(Int32, (split(data[x], "\t")[3] )), indexes)

  S = sparse(users, itens, values, quantUsers, quantItens)

  return S
end

function predict(k, usersTraining, usersTest, meanAndStd, simVector)

  userPredict = Dict()

  fail = 0

  #percorre todos os usuários de teste
  for u in eachindex(usersTest[:,1])

    #Itens que vamos prever
    itens = find(x -> x > 0, usersTest[u,:])

    for i in itens

      #ordena os usuários similares a u
      simUsersSort = sortperm(simVector[u,:], rev=true)

      #pega os usuários que avaliaram i
      usuariosAvalI = find(x -> x > 0, usersTraining[simUsersSort,i])

      #caso a avaliação k seja 0 não consegue prever
      if size(usuariosAvalI)[1] < k || simUsersSort[k] <= 0
        fail += 1
        continue
      end

      num_sum = 0
      don_sum = 0

      for v in simUsersSort[usuariosAvalI[1:k]]
        num_sum += simVector[u,v] * usersTraining[v, i]
        don_sum += abs(simVector[u,v])
      end
      if isnan(num_sum / don_sum)
        fail += 1
        continue
        #userPredict[(u, i)] = 0
      else
        userPredict[(u, i)] = num_sum / don_sum
      end
    end
  end
  println("não consegui prever para $fail itens")
  return userPredict
end

function similarity(u, v, sim_func)
  indexes_u = find(x -> x != 0, u)
  indexes_v = find(x -> x != 0, v)

  indexes = intersect(indexes_u, indexes_v)

  if (size(indexes)[1] < 2)
    return 0
  end

  return sim_func(u[indexes,1], v[indexes,1])
end

function cosine(a, b)
  if isempty(a) || isempty(b)
      return 0;
  end
  d = sqrt(sum(a.^2)) * sqrt(sum(b.^2))
  if d == 0
    return 0
  end
  return sum(a .* b) / d
end

function MAE(predict, test)

  MAE = 0

  for user in keys(predict)
    MAE += abs(predict[user] - test[user[1], user[2]])
  end
  return MAE/length(predict)
end

function RMSE(predict, test)

  RMSE = 0

  for user in keys(predict)
    RMSE += (predict[user] - test[user[1], user[2]])^2
  end

  return sqrt(RMSE/length(predict))
end

function main(k, sim_func, error_func)
  file = open("ml-100k/u.data")

  data = readlines(file)
  dataIndexes = randperm(size(data)[1])
  max = round(Int64, size(dataIndexes)[1] * 0.8)
  testSize = round(Int64, size(dataIndexes)[1] * 0.2)
  i = 0
  #for i in 0:4
  _begin = i * testSize + 1

  #divisão treino e teste
  dataIndex = [_begin:(i * testSize+testSize)...]
  testIndexes = dataIndex
  trainingIndexes = union(dataIndexes[i * testSize+testSize + 1:end],
  dataIndexes[1:i * testSize])

  usersTraining = createDataStruct(trainingIndexes, data)
  usersTest = createDataStruct(testIndexes, data)

  meanAndStd = zeros(size(usersTest)[1], 2)

  #matrix de similaridade
  simVector = zeros(size(usersTraining)[1], size(usersTraining)[1])

  for u in 1: size(usersTraining[:,1])[1]
    meanAndStd[u, 1] = mean(nonzeros(usersTraining[u,:]))
    meanAndStd[u, 2] = std(nonzeros(usersTraining[u,:]))
    for v in u+1: size(usersTraining[:,1])[1]
      sim = similarity(usersTraining[u,:], usersTraining[v,:], sim_func)
      simVector[u, v] = sim
      simVector[v, u] = sim
    end
  end
  userPredict = predict(k, usersTraining, usersTest, meanAndStd,
  simVector)

  println("Error $(error_func(userPredict, usersTest))")
  #end
end

for i in [20,10,80]
  println("Cosine")
  println("$i vizinhos")
  println("MAE")
  main(i, cosine, MAE)
end
