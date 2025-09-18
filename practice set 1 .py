


#एक number को input लेकर check करो कि वो palindrome है या नहीं (बिना str() के)।
# 1
x= 123
temp=x
digit=0
rev=0
while temp>0:
    digit =temp %10
    rev=rev*10+digit
    temp=temp //10
if rev==x:
    print("peali")
else :
    print("not")
#2
x=121
s=str(x)
s=s[::-1]
s=int(s)
if x==s:
    print("pali")
else:
    print("not")

#एक list में से second largest number निकालो।
# 1
l1=[10,20,11,21,30,35]
l2=list(set(l1))
l3=l2.sort(reverse=True)
print(l2[1])

#2
for i in range (len(l1)):
    for j in range (len(l1)-1):
        if l1[j]<l1[j+1]:
            l1[j],l1[j+1]=l1[j+1],l1[j]
print(l1[1])

#एक string में हर character का frequency count निकालो (dict का use करके)।

1#
s="abccda"
d={}
for i in s:
    if i in d:
        d[i]+=1
    else:
        d[i]=1
print(d)

#2
s="asddsad"
d={}
for i in s:
    d[i]=s.count(i)

#Fibonacci series print करो (recursion और loop दोनों से)।

#1
n=10
a,b=0,1
for i in range (n):
    print(a,end=" , ")
    a,b =b,a+b
#2 
def fibo(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else :
        return fibo(n-1)+fibo(n-2)
def p_fibo(n):
    if n>0:
        p_fibo(n-1)
        print(fibo(n-1),end=",")
p_fibo(10)

    
  #  Prime number check करने का program लिखो।

# Prime number check करने का program लिखो
n=4
p=0
for i in range(n):
    
    if n%(i+1)==0:
        p+=1

if n>0:
    if p>2:
        print("not prime")
    else :
        print("prime")


#2

def prime(n,i=1,p=0):
    if i>n:
        if p==2:
            print("prime")
        elif n<=0:
            print("not valid")
        else:
            print("not prime")
        return
    else:
        if n%i==0:
            p+=1
            
        prime(n,i+1,p)
        
prime(-0)

# एक program लिखो जो list में से duplicates remove करे (order maintain करते हुए)।

#1 
l1=[12,32,32,21,14,54,15,12,57,12]
l1=list(set(l1))
l1.sort()
print(l1)

#2
l1=[12,32,32,21,14,54,15,12,57,12]
l2=[]
for i in range(len(l1)):
    for j in range(len(l1)-1):
        if l1[j]>l1[j+1]:
            l1[j],l1[j+1]=l1[j+1],l1[j]
for i in range(len(l1)):
    if l1[i]not in l2:
        l2.append(l1[i])
            
            
print(l2)

#Dictionary में से max value वाला key निकालो
d={'a':10,'b':20,'c':915,'d':13}
l1=[]
for x,i in d.items():
   l1.append(i)
l1.sort(reverse=True)
k={k for k,v in d.items() if v==l1[0]}
print(k)
    
#एक class बनाओ BankAccount जिसमें deposit, withdraw और balance check करने के methods हों।
class BankAccount:
    def __init__(self, initial_balance=0):
        self.balance = initial_balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited: {amount}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds")
        else:
            self.balance -= amount
            print(f"Withdrew: {amount}")

    def get_balance(self):
        return self.balance

    def balance_check(self):
        print(f"Current balance: {self.balance}")

#दो sorted lists merge करके एक sorted list बनाओ (बिना inbuilt sort के)।
l1=[12, 14, 15, 21, 32, 54, 57]
l2=[13,12,17,34,23,89]
l3=l1+l2
l4=[]
for i in range(len(l3)):
    for j in range(len(l3)-1):
        if l3[j]>l3[j+1]:
            l3[j],l3[j+1]=l3[j+1],l3[j]
for i in range(len(l3)):
    if l3[i]not  in l4:
        l4.append(l3[i])
        
print(l4)
