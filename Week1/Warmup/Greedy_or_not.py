n = int(input())
numbers = list(map(int, input().split()))
memo = {}

if len(numbers) != n:
  print("Invalid input")
  exit()

# assuming always the player 1 has first to move

# Memoization

def move(start, end):
  global numbers, memo

  if start == end:
    return numbers[start]

  tupleKey = (start, end)
  
  if tupleKey in memo:
    return memo[tupleKey]
  
  chooseFirst = numbers[start]-move(start+1, end)
  chooseLast = numbers[end]-move(start, end-1)

  memo[tupleKey] = max(chooseFirst, chooseLast)

  return memo[tupleKey]

check = move(0, n-1)

if check > 0:
  print("Player 1 wins")
elif check < 0:
  print("Player 2 wins")
else:
  print("Its a draw")

exit()