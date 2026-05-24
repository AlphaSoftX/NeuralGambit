n = int(input())
numbers = list(map(int, input().split()))
memo = {}

if len(numbers) != n:
  print("Invalid input")
  exit()

# assuming always the player 1 has first to move

# Memoization and Bottom-Up DP

for i in range(0,n):
  memo[(i,i)] = numbers[i]

for length in range(2, n+1):
  for start in range(n-length+1):
    end = length+start-1
    chooseFirst = numbers[start]-memo[(start+1, end)]
    chooseLast = numbers[end]-memo[(start, end-1)]
    memo[(start, end)] = max(chooseFirst, chooseLast)

check = memo[(0, n-1)]

if check > 0:
  print("Player 1 wins")
elif check < 0:
  print("Player 2 wins")
else:
  print("Its a draw")

exit()