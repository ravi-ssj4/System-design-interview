# def solution(arr, M, N):
#     def f(i, bot, days):
#         # base case
#         if i == N - 1:
#             if days == 0:
#                 return M
#             cost = days * arr[i]
#             if bot:
#                 return cost
#             else:
#                 return cost + M

#         # general case
#         cost = days * arr[i]
#         if not bot:
#             cost += M
        
#         notSend = cost + f(i + 1, 0, days)
#         send = cost + f(i + 1, 1, days + 1)

#         return min(send, notSend)

#     return f(0, 0, 0)
    
# def main():
#     print("Enter user inputs")
#     T = int(input())
#     while T:
#         N = int(input())
#         arr = list(map(int, input().strip().split(" ")))
#         M = int(input())
    
#         print(solution(arr, M, N))

#         T = T - 1

# main()

def solution(arr, M, N):
    def f(i, bot):
        # base case
        if i == N - 1:
            return min(M, (i - bot) * arr[i])
            

        # general case
        if bot == -1:
            return M + f(i + 1, i)
        
        keepBot = M + f(i + 1, i)
        notKeepBot = 0 + (i - bot) * arr[i] + f(i + 1, bot)
        
        return min(keepBot, notKeepBot)

    return f(0, -1)
    
def main():
    print("Enter user inputs")
    T = int(input())
    while T:
        N = int(input())
        arr = list(map(int, input().strip().split(" ")))
        M = int(input())
    
        print(solution(arr, M, N))

        T = T - 1

main()