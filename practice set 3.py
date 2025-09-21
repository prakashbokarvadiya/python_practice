#Q1. Find all prime numbers between 1 to 100 (without using any library).

for num in range(2, 101):
    is_prime = True
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        print(num, end=" ")

#Q2. Find the nth Fibonacci number using recursion and memoization.

memo = {}

def fib(n):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]

print(fib(20))

#Q3. Implement a program to rotate a matrix (2D list) 90 degrees clockwise.
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

rotated = list(zip(*matrix[::-1]))
for row in rotated:
    print(row)


#Q4. Find all pairs in a list that sum to a given number.
nums = [2, 7, 11, 15, -2, 9, 4]
target = 9

for i in range(len(nums)):
    for j in range(i+1, len(nums)):
        if nums[i] + nums[j] == target:
            print(nums[i], nums[j])

#Q5. Sort a dictionary by values in descending order.
data = {"a": 10, "b": 30, "c": 20, "d": 25}
sorted_dict = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
print(sorted_dict)


#Q6. Write a function to flatten a nested list.
def flatten(lst):
    result = []
    for i in lst:
        if isinstance(i, list):
            result.extend(flatten(i))
        else:
            result.append(i)
    return result

print(flatten([1, [2, [3, 4], 5], 6]))

#Q7. Check if two strings are anagrams.
s1 = "listen"
s2 = "silent"

if sorted(s1) == sorted(s2):
    print("Anagrams")
else:
    print("Not Anagrams")
#Q8. Find the first non-repeated character in a string.
s = "aabbccdeeff"
for char in s:
    if s.count(char) == 1:
        print("First non-repeated:", char)
        break


#Q9. Generate Pascal’s Triangle up to n rows.
def pascal(n):
    triangle = [[1]]
    for i in range(1, n):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i-1][j-1] + triangle[i-1][j])
        row.append(1)
        triangle.append(row)
    return triangle

for row in pascal(6):
    print(row)


#Q10. Implement binary search (without recursion).
def binary_search(arr, target):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

print(binary_search([1, 3, 5, 7, 9], 7))


#Q11. Find longest word in a sentence.
sentence = "Artificial Intelligence and Machine Learning"
words = sentence.split()
longest = max(words, key=len)
print("Longest word:", longest)


#Q12. Find duplicates in a list using dictionary.
nums = [1, 2, 3, 4, 2, 5, 3, 6, 1]
freq = {}
for n in nums:
    freq[n] = freq.get(n, 0) + 1
duplicates = [k for k, v in freq.items() if v > 1]
print("Duplicates:", duplicates)


#Q13. Reverse words in a sentence.
sentence = "Python is very powerful"
reversed_sentence = " ".join(sentence.split()[::-1])
print(reversed_sentence)


#Q14. Find intersection of two lists.
a = [1, 2, 3, 4, 5]
b = [4, 5, 6, 7]
intersection = [x for x in a if x in b]
print("Intersection:", intersection)



#Q15. Implement stack using a class.
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop() if not self.is_empty() else None

    def peek(self):
        return self.items[-1] if not self.is_empty() else None

    def is_empty(self):
        return len(self.items) == 0

s = Stack()
s.push(10)
s.push(20)
print(s.pop())
print(s.peek())