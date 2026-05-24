n = int(input())
numbers = list(map(int, input().split()))

if len(numbers) != n:
  print("Invalid input")
  exit()

# assuming always the player 1 has first to move

def move(start, end):
  global numbers

  if start == end:
    return numbers[start]
  
  chooseFirst = numbers[start]-move(start+1, end)
  chooseLast = numbers[end]-move(start, end-1)

  return max(chooseFirst, chooseLast)

check = move(0, n-1)

if check > 0:
  print("Player 1 wins")
elif check < 0:
  print("Player 2 wins")
else:
  print("Its a draw")

exit()