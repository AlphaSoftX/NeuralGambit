import numpy as np

n = int(input())
numbers = list(map(int, input().split()))

if len(numbers) != n:
  print("Invalid input")
  exit()

# assuming always the player 1 has first to move

# 1D Array Memoization and Bottom-Up DP --> Using NumPy
nums = np.array(numbers)
memo = nums.copy()

for length in range(2, n + 1):
  starts = np.arange(n - length + 1)
  ends = starts + length - 1
  chooseFirst  = nums[starts] - memo[starts + 1]
  chooseLast = nums[ends]   - memo[starts]
  memo[starts] = np.maximum(chooseFirst, chooseLast)

check = memo[0]

if check > 0:
  print("Player 1 wins")
elif check < 0:
  print("Player 2 wins")
else:
  print("Its a draw")

exit()