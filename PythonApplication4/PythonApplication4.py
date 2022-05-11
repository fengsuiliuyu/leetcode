﻿'''
leetcode 2257 未被守卫的格子数
from collections import defaultdict
def countUnguarded(m, n, guards, walls):
        allpoints = []
        def xbright(index,pointlist):
            cindex = 0
            start = 0
            brightpoint = []
            pointlist.sort(key = lambda x:x[2])
            for i in pointlist:
                end = i[2]
                if i[0] == 1 and end > start:
                    for j in range(start,end):
                        brightpoint.append(index+j*1e5)
                        cindex = 1
                elif i[0] == 1:
                    cindex = 1
                elif cindex == 1 and end>start:
                    for j in range(start,end):
                        brightpoint.append(index+j*1e5)
                        cindex = 0
                else:
                    cindex = 0
                start = i[2] + 1
            if pointlist[-1][0] == 1:
                for j in range(start,n):
                    brightpoint.append(index+j*1e5)
            return brightpoint
        def ybright(index,pointlist):
            cindex = 0
            start = 0
            brightpoint = []
            pointlist.sort(key = lambda x:x[1])
            for i in pointlist:
                end = i[1]
                if i[0] == 1 and end > start:
                    for j in range(start,end):
                        brightpoint.append(j+index*1e5)
                        cindex = 1
                elif i[0] == 1:
                    cindex = 1
                elif cindex == 1 and end>start:
                    for j in range(start,end):
                        brightpoint.append(j+index*1e5)
                        cindex = 0
                else:
                    cindex = 0
                start = i[1] + 1
            if pointlist[-1][0] == 1:
                for j in range(start,m):
                    brightpoint.append(j+index*1e5) 
            return brightpoint
        newguards  = [[1]+i for i in guards]
        newwalls = [[2] + i for i in walls]
        xallpointlist = defaultdict(list)
        yallpointlist = defaultdict(list)
        rangeall = newguards + newwalls
        for i in rangeall:
            xallpointlist[i[1]].append(i)
            yallpointlist[i[2]].append(i)
        for i in xallpointlist:
            allpoints += xbright(i,xallpointlist[i])
        for i in yallpointlist:
            allpoints += ybright(i,yallpointlist[i])
        anwser = set(allpoints)
        return m*n - len(anwser) - len(guards) - len(walls)
a = countUnguarded(2,7,[[1,5],[1,1],[1,6],[0,2]],[[0,6],[0,3],[0,5]])
print(a)
'''

'''
leetcode 2262 字符串总引力

class Solution:
    def appealSum(s: str) -> int:
        anwser , sum_g ,last =0,0, [-1]*26
        for i in range(len(s)):
            c = 0
            c = ord(s[i]) - ord('a')
            sum_g += i - last[c]
            anwser += sum_g
            last[c] = i
        return anwser
    a = appealSum('abbca')
    print(a)

'''





'''
leetcode 2261 含最多k个可整除元素的子数组

from collections import defaultdict
class Solution:
    def countDistinct(self, nums: list[int], k: int, p: int) -> int:
        lenth = len(nums)
        numsindex = []
        for i in nums:
            if i%p == 0:
                numsindex.append(i)
        anwserdict = defaultdict(list)
        anwser = 0
        for i in range(lenth):
            start = 0
            end = lenth - i
            temp = 0
            if i == 0:
                if len(numsindex) <= k:
                    anwser += 1
            else:
                for j in range(lenth - i): 
                    if nums[j]%p == 0:
                        temp += 1 #开始的子数组的整除数量
                if temp <= k:
                    anwserdict[tuple(nums[start:end])].append(i)
                for j in range(i):
                    if nums[start]%p == 0:
                        temp -= 1
                    if nums[end]%p == 0:
                        temp += 1
                    start += 1
                    end += 1
                    if temp <= k:
                        anwserdict[tuple(nums[start:end])].append(i)
        anwser += len(anwserdict.keys())
        return anwser
    a = countDistinct(0,[2,3,3,2,2],2,2)
    print(a)
'''
                    
       


'''
leetcode 2227 加密解密字符串

from collections import defaultdict
class Encrypter:
    def __init__(self, keys: list[str], values: list[str], dictionary: list[str]):
        self.keys = keys
        self.cnt1 = {}
        self.dictionary = dictionary
        for i in range(len(keys)):
            self.cnt1[keys[i]] = i
            
        self.values = values
        self.cnt2 = defaultdict(list)
        for i in range(len(values)):
            self.cnt2[values[i]].append(i)
        self.d = defaultdict(set)
        for i in dictionary:
            temp = ''
            index = 0
            for j in i:
                 index += 1
                 temp += j
                 self.d[index].add(temp)
        self.al = {}
    def encrypt(self, word1: str) -> str:
        ret = ""
        for n in word1:
            ret += self.values[self.cnt1[n]]
        return ret
        
    def decrypt(self, word2: str) -> int:
        if word2 in self.al:
            return self.al[word2]
        lst = [""]
        p = 0
        index = 0
        ans = 0
        while p < len(word2):
            cur = word2[p:p+2]
            index += 1
            if cur in self.cnt2:
                nlst = []
                for newp in self.cnt2[cur]:
                    nlst += [n + self.keys[newp] for n in lst if (n + self.keys[newp]) in self.d[index]]
                lst = nlst
            else:
                return 0
            p += 2
        
        for i in lst:
            if i in self.dictionary:
                ans += 1
        self.al[word2] = ans
        return ans
''' 

'''
leetcode 2258 逃离火灾

from copy import deepcopy
class Solution:
    def maximumMinutes(self, grid: list[list[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        firstpointdict = {}
        firstfirepoint = []
        for x in range(m):
            for y in range(n):
                if grid[x][y] == 1:
                    firstpointdict[(x,y)] = 1
                    firstfirepoint.append((x,y))
                elif grid[x][y] == 2:
                    firstpointdict[(x,y)] = 2
                else:
                    firstpointdict[(x,y)] = 0
        def fire_can(t):
            nonlocal firstfirepoint
            firepoint = []
            firepoint += firstfirepoint
            pointdict = deepcopy(firstpointdict)
            for i in range(t):
                temp = []
                for j in firepoint:
                    for x,y in ((j[0]+1,j[1]),(j[0]-1,j[1]),(j[0],j[1]+1),(j[0],j[1]-1)):
                        if 0 <= x and x <= m-1 and 0 <= y and y <= n-1 and pointdict[(x,y)] == 0:
                            temp.append((x,y))
                            pointdict[(x,y)] = 1
                firepoint = []
                firepoint += temp
            if pointdict[(0,0)] == 1:
                return 1
            step = [(0,0)]
            pointdict[(0,0)] = -1
            for i in range(m*n):
                temp = []
                steptemp = []
                for j in step:
                    for x,y in ((j[0]+1,j[1]),(j[0]-1,j[1]),(j[0],j[1]+1),(j[0],j[1]-1)):
                        if 0 <= x and x <= m-1 and 0 <= y and y <= n-1 and pointdict[(x,y)] == 0:
                            steptemp.append((x,y))
                            pointdict[(x,y)] = -1
                step = []
                step += steptemp
                if pointdict[(m-1,n-1)] == -1:
                    return 0
                for j in firepoint:
                    for x,y in ((j[0]+1,j[1]),(j[0]-1,j[1]),(j[0],j[1]+1),(j[0],j[1]-1)):
                        if 0 <= x and x <= m-1 and 0 <= y and y <= n-1 and pointdict[(x,y)] != 1 and pointdict[(x,y)] != 2:
                            temp.append((x,y))
                            pointdict[(x,y)] = 1
                firepoint = []
                firepoint += temp
                for i in temp:
                    if i in step:
                        step.remove(i)
            return 1
        time = m*n//2
        start = 0
        end = m*n
        while True:
            if time == m*n:
                return 1000000000
            if fire_can(time) == 0:
                if fire_can(time+1) == 1:
                    return time
                else:
                    start = time
                    time += (end-time+1)//2
                    continue
            if time == 0:
                return -1
            else:
                end = time
                time -= (time-start+1)//2
    a = maximumMinutes(0,[[0,2,0,0,0,0,0],[0,0,0,2,2,1,0],[0,2,0,0,1,2,0],[0,0,2,2,2,0,2],[0,0,0,0,0,0,0]])
    print(a)
'''

'''
leetcode 6059 检查是否有合法括号字符串路径
from collections import defaultdict
class Solution:
    def hasValidPath(self, grid: list[list[str]]) -> bool:
        def judgement(j,pointselect):
            showindex = 0
            if pointselect == 1:
                return [False]
            elif pointselect == -1:
                x,y = j[0]+1,j[1]
                if x<= m-1 and y <= n-1:
                    if grid[x][y] == '(' and j[2] <= (m+n-1)//2-1:
                        newj = [x,y,j[2]+1,j[3]]
                        showindex = 1
                    elif grid[x][y] == ')' and j[3] <= j[2] - 1:
                        newj = [x,y ,j[2],j[3]+1]
                        showindex = 1
            if showindex == 1:
                pointselect = 0
                return [newj,pointselect]
            else:
                x,y = j[0],j[1]+1
                if x<= m-1 and y <= n-1:
                    if grid[x][y] == '(' and j[2] <= (m+n-1)//2-1:
                        newj = [x,y,j[2]+1,j[3]]
                        showindex = 1
                    if grid[x][y] == ')' and j[3] <= j[2] - 1:
                        newj = [x,y ,j[2],j[3]+1]
                        showindex = 1
                if showindex == 1:
                    pointselect = 1
                    return [newj,pointselect]
                else:
                    return [False]
        pathdict = {}
        m = len(grid)
        n = len(grid[0])
        visited = [0,0,1,0]
        if grid[0][0] == ')' or grid[m-1][n-1] == '(' or (m+n)%2 == 0:
            return False
        pathdict[0] = [[0,0,1,0],-1]
        index = 0
        while True:
            newpoint = judgement(pathdict[index][0],pathdict[index][1])
            if newpoint == [False]:
                index -= 1
                if index == -1:
                    return False
                continue
            pathdict[index][1] = newpoint[1]
            pathdict[index+1] = [newpoint[0],-1]
            if newpoint[0] in visited:
                continue
            visited.append(newpoint[0])
            index += 1
            if index == m+n-2:
                return True
    a = hasValidPath(0,[["(","(","("],[")","(",")"],["(","(",")"],["(","(",")"]])
    print(a)
'''

'''               
Leetcode 2266 统计打字方案数

class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        nums = [1,2,4]
        def firstchoice(n):
            nonlocal nums
            if n < len(nums):
                return nums[n]
            for i in range(len(nums),n+1):
                nums.append(nums[i-1]+nums[i-2]+nums[i-3])
            return nums[n]
        secondnums = [1,2,4,8]
        def secondchioce(n):
            nonlocal secondnums
            if n < len(secondnums):
                return secondnums[n]
            for i in range(len(secondnums),n+1):
                secondnums.append(secondnums[i-1]+secondnums[i-2]+secondnums[i-3]+secondnums[i-4])
            return secondnums[n]
        startindex = 0
        index = 0
        anwser = 1
        while True:
            if startindex == len(pressedKeys) - 1:
                break
            if pressedKeys[startindex + 1] == pressedKeys[startindex]:
                index += 1
                startindex += 1
                if startindex == len(pressedKeys) - 1:
                    if pressedKeys[startindex] == '7' or pressedKeys[startindex] ==  '9':
                        anwser *= secondchioce(index)
                    else:
                        anwser *= firstchoice(index)
                        break
            else:
                if pressedKeys[startindex] == '7' or pressedKeys[startindex] == '9':
                    anwser *= secondchioce(index)
                else:
                    anwser *= firstchoice(index)
                startindex += 1
                index = 0
        anwser = anwser%(int(1e9+7))
        return anwser
'''

'''
leetcode 6643 统计包含每个点的矩形数目 

from collections import defaultdict
from bisect import bisect_left

class Solution:
    def countRectangles(self, rectangles: list[list[int]], points: list[list[int]]) -> list[int]:
        rectangles.sort(key = lambda x:(x[1],-x[0]),reverse = True)
        anwser = [0]*len(points)
        recdict = defaultdict(list)
        for i in rectangles:
            recdict[i[1]].append(i[0])
        for i,id in sorted(zip(points,range(len(points))),key = lambda x:-x[0][1]):
            tempanwser = 0
            for j in recdict:
                if j <  i[1]:
                    break
                tempanwser += len(recdict[j])- bisect_left(recdict[j],i[0])
            anwser[id] = tempanwser
        return anwser
    a = countRectangles(0,[[1,1],[2,2],[3,3]],[[1,3],[1,1]])
    print(a)
'''
