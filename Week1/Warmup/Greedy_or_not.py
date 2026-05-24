n = int(input())
numbers = list(map(int, input().split()))

if len(numbers) != n:
  print("Invalid input")
  exit()

# assuming always the player 1 has first to move

# 1D Array Memoization and Bottom-Up DP
memo = numbers.copy()

for length in range(2, n+1):
  for start in range(n-length+1):
    end = length+start-1
    chooseFirst = numbers[start]-memo[start+1]
    chooseLast = numbers[end]-memo[start]
    memo[start] = max(chooseFirst, chooseLast)

check = memo[0]

if check > 0:
  print("Player 1 wins")
elif check < 0:
  print("Player 2 wins")
else:
  print("Its a draw")

exit()