'''
技巧稳固  快速排序实现

def quick_sort(lists,key = lambda x:x,reverse = False):
    if len(lists) <= 1:
        return lists
    start = 0
    end = len(lists)-1
    temp = lists[start]
    while start < end:
        while key(lists[end]) >= temp and end>start:
            end -= 1
        lists[start] = lists[end]
        while temp >= key(lists[start]) and end>start:
            start += 1
        lists[end] = lists[start]
    lists1 = quick_sort(lists[0:start],key,reverse)
    lists2 = quick_sort(lists[start+1:],key,reverse)
    lists = lists1 + [temp] + lists2
    if reverse == True:
        lists = reversed(lists)
    return lists
a= quick_sort([5,3,6,3,8,13,4,31,4,5])
print(a)
'''

'''
leetcode 4 寻找两个正序数组的中位数
'''


'''
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

'''
leetcode 2203 得到要求路径的最小带权子图

from collections import defaultdict
class Solution:
    def minimumWeight(self, n: int, edges: list[list[int]], src1: int, src2: int, dest: int) -> int:
        pointdict = defaultdict(list)
        for i in edges:
            pointdict[i[0]].append([i[1],i[2]])
        pointdict1 = defaultdict(list)
        for i in edges:
            pointdict1[i[1]].append([i[0],i[2]])
        def search_allpath(pointdict,point_1):
            newpoint = [[point_1,0]]
            passpointdict = [10000000000]*n
            passpointdict[point_1] = 0
            while newpoint:
                pointlist = []
                for i in newpoint:
                    if i[1] > passpointdict[i[0]]:
                        continue
                    for j in pointdict[i[0]]:
                        temp = i[1]+j[1]
                        if passpointdict[j[0]] == 0 or passpointdict[j[0]] > temp:
                            passpointdict[j[0]] = temp
                            pointlist.append([j[0],temp])
                newpoint = pointlist
            passpointdict[point_1] = 0
            return passpointdict
        a = search_allpath(pointdict,src1)
        b = search_allpath(pointdict,src2)
        c = search_allpath(pointdict1,dest)
        ans = min(sum(d) for d in zip(a, b, c))
        return ans if ans < 10000000000 else -1
    a = minimumWeight(0,6,[[0,2,2],[0,5,6],[1,0,3],[1,4,5],[2,1,1],[2,3,3],[2,3,4],[3,4,2],[4,5,1]],0,1,5)
    print(a)

'''

'''
leetcode 2209 用地毯覆盖过后的最少白色砖块

class Solution:
    def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
        DP = {}
        index = 0
        for i in range(carpetLen):
            if floor[i] == '1':
                index += 1
            DP[(i,0)] = index
            DP[(i,1)] = 0
        def dp(lenth,m):
            nonlocal DP
            if (lenth,m) in DP:
                return DP[(lenth,m)]
            if m*carpetLen >= lenth+1:
                DP[(lenth,m)] = 0
                return 0
            if m == 0:
                index = 0
                for i in floor[0:lenth+1]:
                    if i == '1':
                        index += 1
                DP[(lenth,m)] = index
                return index
            else:
                if floor[lenth] == '0':
                    DP[(lenth,m)] = dp(lenth-1,m)
                    return DP[(lenth,m)]
                else:
                    DP[(lenth,m)] = min(dp(lenth-1,m)+(floor[lenth]=="1"),dp(lenth-carpetLen,m-1))
                    return DP[(lenth,m)]
        return dp(len(floor)-1,numCarpets)
'''

'''
Leetcode 2122 还原原数组
没解决
from collections import defaultdict
class Solution:
    def recoverArray(self, nums: list[int]) -> list[int]:
        nums.sort()
        newnums = [nums[0]]
        temp = nums[0]
        for i in nums[1:]:
            if i != temp:
                newnums.append(i)
                temp = i
        numsdict = defaultdict(list)
        for i in range(len(newnums)-1):
            for j in range(i+1,len(newnums)):
                numsdict[newnums[j]-newnums[i]].append([i,j])
        for i in numsdict:
            if len(numsdict[i]) == len(newnums)//2 and i%2 == 0:
                anwsernums = i
                break
        allnumber = []
        for i in numsdict[anwsernums]:
            allnumber.append(i[0])
        anwser = []
        for i in allnumber:
            for j in nums:
                if j == nums[i]:
                    anwser.append(j+anwsernums//2)
        return anwser
    a = recoverArray(0,[11,6,3,4,8,7,8,7,9,8,9,10,10,2,1,9])
    print(a)
'''

'''
leetcode 5.14双周赛
2270 分割数组的方案数
class solution:
    def waysToSplitArray(self, nums: list[int]) -> int:
        start = nums[0]
        end = 0
        for i in nums:
            end += i
        end -= start
        if len(nums) == 2:
            if nums[1] <= nums[0]:
                return 1
            else:
                return 0
        anwser = 0
        index = 1
        for i in range(len(nums)-2):
            if start >= end:
                anwser += 1
            start += nums[index]
            end -= nums[index]
            index += 1
        return anwser
    a = waysToSplitArray(0,[2,3,1,0])
    print(a)
'''

'''
Leetcode 2271 毯子覆盖最多的白色砖块数

class Solution:
    def maximumWhiteTiles(self, tiles: list[list[int]], carpetLen: int) -> int:
        if carpetLen == 1:
            return 1
        lenth = len(tiles)
        tiles.sort(key = lambda x : x[0])
        nums = []
        blank = []
        for i in range(lenth-1):
            nums.append(tiles[i][1]-tiles[i][0]+1)
            blank.append(tiles[i+1][0]-tiles[i][1]-1)
        nums.append(tiles[-1][1]-tiles[-1][0]+1)
        blank.append(1e9)
        def change(index,endindex,left,show):
            number = 0
            if index == 0:
                lenth = 0
                endindex = 0
                lenth = carpetLen
                while True:
                    if lenth < nums[endindex]:
                        left = nums[endindex] - lenth
                        show = 1
                        number += lenth
                        break
                    elif lenth == nums[endindex]:
                        left = blank[endindex]
                        number += lenth
                        show = 0
                        break
                    else:
                        lenth -= nums[endindex]
                        number += nums[endindex]
                        if lenth < blank[endindex]:
                            left = blank[endindex] - lenth
                            show = 0
                            break
                        elif lenth == blank[endindex]:
                            left = nums[endindex+1]
                            show = 1
                            endindex += 1
                            break
                        else:
                            lenth -= blank[endindex]
                            endindex += 1
                            show = 1
            else:
                lenth = tiles[index][0] - tiles[index-1][0]       
                if show == 1:
                    while True:
                        if lenth < left:
                            left -= lenth
                            number += lenth
                            break
                        elif lenth == left:
                            left = blank[endindex]
                            number += lenth
                            show = 0
                            break
                        else:
                            lenth -= left
                            number += left
                            if lenth < blank[endindex]:
                                left = blank[endindex] - lenth
                                show = 0
                                break
                            elif lenth == blank[endindex]:
                                endindex += 1
                                left = nums[endindex]
                                break
                            else:
                                lenth -= blank[endindex]
                                endindex += 1
                                left = nums[endindex]
                else:
                    while True:
                        if lenth < left:
                            left = left - lenth
                            break
                        elif lenth == left:
                            endindex += 1
                            left = nums[endindex]
                            show = 1
                            break
                        else:
                            endindex += 1
                            lenth -= left
                            if lenth < nums[endindex]:
                                left = nums[endindex] - lenth
                                show = 1
                                number += lenth
                                break
                            elif lenth == nums[endindex]:
                                left = blank[endindex]
                                number += lenth
                                break
                            else:
                                left = blank[endindex]
                                lenth -= nums[endindex]
                                number += nums[endindex]
            index += 1
            return index,endindex,left,show,number
        index = 0
        endindex = 0
        left = 0
        show = 0
        number = 0
        anwser = 0
        allanwser = []
        for i in range(lenth):
            index,endindex,left,show,number = change(index,endindex,left,show)
            if i == 0:
                anwser = number
            else:
                anwser += number - nums[i-1]
            allanwser.append(anwser)
        allanwser.sort()
        return allanwser[-1]
'''

'''
Leetcode 2272 最大波动的子字符串

from string import ascii_lowercase
from math import inf
class Solution:
    def largestVariance(self, s: str) -> int:
        ans = 0
        for a in ascii_lowercase:
            for b in ascii_lowercase:
                if a == b:
                    continue
                f,g = 0,-inf
                for i in s:
                    if i == a:
                        f += 1
                        g += 1
                    if i == b:
                        f = f-1
                        g = f
                        if f < 0:
                            f = 0
                    if g > ans:
                        ans = g
        return ans
'''

'''
leetcode 668 乘法表中第k小的数

class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        if m > n:
            temp = n
            n = m
            m = temp
        def search(i):
            anwser = 0
            for j in range(1,m+1):
                anwser += min(i//j,n)
            return anwser
        start = 1
        end = m+1
        while start < end:
            temp = (start+end)//2
            judge = search(temp*n)
            if judge >k:
                end = temp
            elif judge == k:
                return temp*n
            else:
                start = temp + 1
        allnumber = []
        for i in range(start,m+1):
            for j in range((start-1)*n//i+1,start*n//i+1):
                allnumber.append(j*i)
        allnumber.sort()
        return allnumber[k-1-search((start-1)*n)]
    a = findKthNumber(0,9,9,81)
    print(a)
'''



'''
Leetcode 6066 统计区间中的整数数目

class CountIntervals:

    def __init__(self):
        self.left = []
        self.right = []
        self.num = 0

    def add(self, left: int, right: int) -> None:
        leftindex = bisect.bisect_left(self.left,left)
        rightindex = bisect.bisect_right(self.right,right)
        leftindex2 = bisect.bisect_left(self.right,left)
        rightindex2 = bisect.bisect_right(self.left,right)
        if leftindex2 == len(self.left):
            self.left.append(left)
            self.right.append(right)
            self.num += right -left + 1
        elif rightindex2 == 0:
            self.left.insert(0,left)
            self.right.insert(0,right)
            self.num += right -left + 1
        elif leftindex == leftindex2:
            if rightindex == rightindex2:
                total = 0
                leftdelete = []
                rightdelete = []
                for i in range(leftindex,rightindex):
                    total += self.right[i]-self.left[i]+1 
                    leftdelete.append(i)
                    rightdelete.append(i)
                change = right - left +1 -total
                self.num += change
                for i in range(rightindex-1,leftindex-1,-1):
                    self.left.pop(i)
                    self.right.pop(i)
                self.left.insert(leftindex,left)
                self.right.insert(leftindex,right)
            else:
                right = self.right[rightindex]
                rightindex += 1
                total = 0
                leftdelete = []
                rightdelete = []
                for i in range(leftindex,rightindex):
                    total += self.right[i]-self.left[i]+1 
                    leftdelete.append(i)
                    rightdelete.append(i)
                change = right - left +1 -total
                self.num += change
                for i in range(rightindex-1,leftindex-1,-1):
                    self.left.pop(i)
                    self.right.pop(i)
                self.left.insert(leftindex,left)
                self.right.insert(leftindex,right)
        else:
            if rightindex == rightindex2:
                left = self.left[leftindex2]
                leftindex -= 1
                total = 0
                leftdelete = []
                rightdelete = []
                for i in range(leftindex,rightindex):
                    total += self.right[i]-self.left[i]+1 
                    leftdelete.append(i)
                    rightdelete.append(i)
                change = right - left +1 -total
                self.num += change
                for i in range(rightindex-1,leftindex-1,-1):
                    self.left.pop(i)
                    self.right.pop(i)
                self.left.insert(leftindex,left)
                self.right.insert(leftindex,right)
            else:
                left = self.left[leftindex2]
                right = self.right[rightindex]
                rightindex += 1
                leftindex -= 1
                total = 0
                leftdelete = []
                rightdelete = []
                for i in range(leftindex,rightindex):
                    total += self.right[i]-self.left[i]+1 
                    leftdelete.append(i)
                    rightdelete.append(i)
                change = right - left +1 -total
                self.num += change
                for i in range(rightindex-1,leftindex-1,-1):
                    self.left.pop(i)
                    self.right.pop(i)
                self.left.insert(leftindex,left)
                self.right.insert(leftindex,right)

    def count(self) -> int:
        return self.num
'''

'''
leetcdoe 2281 巫师的总力量和

import bisect
class Solution:
    def totalStrength(self, strength: list[int]) -> int:
        minnum = []
        power = []
        for i in strength:
            time = bisect.bisect_left(minnum,i)
            if time == len(minnum):
                minnum.append(i)
                index = 0
                for j in power:
                    j[1] += j[2]*i
                    j[0] += j[1]*minnum[index]
                    index += 1
                power.append([i*i,i,1])
            else:
                minnum.insert(time,i)
                minnum = minnum[0:time+1]
                index = 0
                for j in power:
                    j[1] += j[2]*i
                    j[0] += j[1]*minnum[min(index,time)]
                    index += 1
                power.append([i*i,i,1])
                temppower =[0,0,0]
                for j in range(time,len(power)):
                    temppower[0] += j[0]
                    temppower[1] += j[1]
                    temppower[2] += j[2]
                power.insert(time,temppower)
                power = power[0:time+1]
            anwser = 0
            for  i in power:
                anwser += i[0]
        return anwser
'''

'''
leetcode 675 为高尔夫比赛砍树

from collections import defaultdict
class Solution:
    def cutOffTree(self, forest: list[list[int]]) -> int:
        m = len(forest)
        n = len(forest[0])
        numberdict = {}
        trees = []
        for i in range(len(forest)):
            for j in range(len(forest[0])):
                if forest[i][j] > 1:
                    trees.append(forest[i][j])
                    numberdict[forest[i][j]] = (i,j)
        trees.sort()
        start = (0,0)
        index = 0
        anwser = 0
        tempanwser = 0
        def judge(start,end):
            if start == end:
                return 0
            time = 0
            newpoint = []
            startpoint = [start]
            visited ={start}
            while startpoint:
                time += 1
                for i in startpoint:
                    for j,k in [(i[0]-1,i[1]),(i[0]+1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1)]:
                        if 0<= j < m and 0<=k <n and forest[j][k] >= 1 and (j,k) not in visited:
                            if (j,k) == end:
                                return time
                            newpoint.append((j,k))
                            visited.add((j,k))
                startpoint = newpoint
                newpoint = []
            return -1
        for i in range(len(trees)):
            end = numberdict[trees[index]]
            tempanwser = judge(start,end)
            if tempanwser == -1:
                return -1
            anwser += tempanwser
            start = end
            index += 1
        return anwser
A = Solution
a = A.cutOffTree(A,[[54581641,64080174,24346381,69107959],[86374198,61363882,68783324,79706116],[668150,92178815,89819108,94701471],[83920491,22724204,46281641,47531096],[89078499,18904913,25462145,60813308]])
print(a)
'''


'''
leetcode 464 我能赢吗
import functools
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        @functools.lru_cache(maxsize = None)
        def dfs(usedNumbers: int, currentTotal: int) -> bool:
            for i in range(maxChoosableInteger):
                if (usedNumbers >> i) & 1 == 0:
                    if currentTotal + i + 1 >= desiredTotal or not dfs(usedNumbers | (1 << i), currentTotal + i + 1):
                        return True
            return False

        return (1 + maxChoosableInteger) * maxChoosableInteger // 2 >= desiredTotal and dfs(0, 0)
    a = canIWin(0,18,70)
    print(a)
'''

'''
leetcode 467 环绕字符串中唯一的字符串

class Solution:
    def findSubstringInWraproundString(self, p: str) -> int: 
        right = 0
        allstr = set()
        strdict = collections.defaultdict(int)
        anwser = 0
        for i in range(len(p)):
            if i <= right and i != 0:
                if right-i+1 > strdict[p[i]]:
                    strdict[p[i]] = right - i + 1
            else:
                right = i
                if right == len(p)-1:
                    if 1 > strdict[p[i]]:
                        strdict[p[i]] = 1
                    break
                while right <len(p) and (ord(p[right+1]) ==  ord(p[right])+1 or (p[right+1]=='a'and p[right]=='z')):
                    right += 1
                    if right == len(p)-1:
                        break
                if right-i+1 > strdict[p[i]]:
                    strdict[p[i]] = right-i+1
        for j in strdict:
            anwser += strdict[j]
        return anwser
'''
'''
from collections import defaultdict
class BookMyShow:
    def __init__(self, n: int, m: int):
        self.allseats = defaultdict(list)
        self.m = m
        self.numberdict = defaultdict(int)
        
    def gather(self, k: int, maxRow: int) -> list[int]:
        m = self.m
        for i in range(0,maxRow+1):
            if k > m-self.numberdict[i]:
                continue
            if self.allseats[i] == []:
                self.allseats[i].append([0,k-1])
                self.numberdict[i] += k
                return [i,0]
            temp = 0
            index = 0
            for j in self.allseats[i]:
                if j[0] >= temp + k:
                    self.allseats[i].insert(index,[temp,temp+k-1])
                    self.numberdict[i] += k
                    return [i,temp]
                temp = j[1]+1
                index += 1
            if m >= temp + k:
                self.allseats[i].append([temp,temp+k-1])
                self.numberdict[i] += k
                return [i,temp]
        return []
    
    def scatter(self, k: int, maxRow: int) -> bool:
        m = self.m
        lenth = 0
        for i in range(0,maxRow+1):
            lenth += self.numberdict[i]
            if (i+1)*m-lenth > k:
                break
        if (maxRow+1)*m - lenth <  k:
            return False
        for i in range(0,maxRow+1):
            if k >= m-self.numberdict[i]:
                k -= m -self.numberdict[i]
                self.numberdict[i] = m
            else:
                if k == 0 :
                    break
                temp = 0
                while self.allseats[i] != [] and k >self.allseats[i][0][0]-temp:
                    k -= self.allseats[i][0][0]-temp
                    temp = self.allseats[i][0][1]+1
                    self.allseats[i].pop(0)
                if self.allseats[i] == []:
                    self.allseats[i].append([0,temp + k-1])
                else:
                    self.allseats[i].insert(0,[0,temp + k-1])
                self.numberdict[i] += k
                k = 0
                break
        return True
a = BookMyShow(18,48)
print(a.scatter(24,13))
print(a.scatter(12,5))
print(a.gather(12,5))
'''

'''
class Solution:
    def totalSteps(self, nums: list[int]) -> int:
        def judge(s,show):
            if len(s) <= 1:
                return show
            big = s[0]
            temp = s[0]
            index = 1
            news = []
            for i in s:
                if i>= temp:
                    news.append(i)
                temp = i
            return judge(news,show+1)
        start = 0
        index = 1
        anwser = 0
        while index < len(nums):
            if nums[index] <nums[start]:
                index += 1
            else:
                anwser = max(anwser,judge(nums[start:index],0))
                start = index
                index += 1
        anwser = max(anwser,judge(nums[start:index],0))
        return anwser
    a = totalSteps(0,[10,1,2,3,4,5,6,1,2,3])
    print(a)
'''

'''
剑指 Offer II 114. 外星文字典

import collections
import copy
class Solution:
    def alienOrder(self, words: list[str]) -> str:
        def judge(a,b):
            m = len(a)
            n = len(b)
            i,j = 0,0
            while i<m and j <n and a[i] == b[j]:
                i+= 1
                j += 1
            if i ==m or j == n:
                if len(a) > len(b):
                    return False
                return None
            else:
                return (a[i],b[j])
        worddictl = collections.defaultdict(list)
        allword = set()
        for i in range(len(words)-1):
            temp = judge(words[i],words[i+1])
            if temp == False:
                return ''
            if temp != None:
                if temp[0] not in worddictl[temp[1]]:
                    worddictl[temp[1]].append(temp[0])
        for i in words:
            for j in i:
                allword.add(j)
            if len(allword) == 26:
                break
        def find(worddictl,allword):
            newallword = copy.deepcopy(allword)
            for i in worddictl:
                allword.remove(i)
            if not allword:
                return (False,False,False)
            string = ''.join(list(allword))
            for i in worddictl:
                for j in list(allword):
                    if j in worddictl[i]:
                        worddictl[i].remove(j)
            deleteword = []
            for i in worddictl:
                if worddictl[i] == []:
                    deleteword.append(i)
            for i in deleteword:
                worddictl.pop(i)
            for i in list(allword):
                newallword.remove(i)
            return (string,worddictl,newallword)
        allstr = []
        while allword:
            string,worddictl,allword = find(worddictl,allword)
            if string == False:
                return ''
            allstr.append(string)
        return ''.join(allstr)
    a = alienOrder(0,["ac","ab","zc","zb"])
    print(a)
'''

'''
leetcode 699 掉落的方块
class Solution:
    def fallingSquares(self, positions: list[list[int]]) -> list[int]:
        height = []
        allheight = []
        interval= []
        for i in positions:
            if not interval:
                interval.append([i[0],i[0]+i[1]])
                height = [i[1]]
                allheight = [i[1]]
            else:
                index = 0
                while index < len(interval) and i[0] >= interval[index][1] :
                    index += 1
                if index == len(interval):
                    allheight.append(i[1])
                    height.append(max(height[-1],i[1]))
                    interval.append([i[0],i[0]+i[1]])
                else:
                    if i[0] > interval[index][0]:
                        interval.insert(index,[interval[index][0],i[0]])
                        allheight.insert(index,allheight[index])
                        maxheight = allheight[index]
                        index += 1
                        while index < len(interval) and (i[0]+i[1]) >= interval[index][1]:
                            maxheight = max(maxheight,allheight[index])
                            allheight.pop(index)
                            interval.pop(index)
                        if index == len(interval):
                            allheight.append(maxheight + i[1])
                            interval.append([i[0],i[0]+i[1]])
                            height.append(max(height[-1],maxheight+i[1]))
                        elif (i[0]+i[1]) > interval[index][0]:
                            maxheight = max(maxheight,allheight[index])
                            allheight.insert(index,maxheight+i[1])
                            interval.insert(index,[i[0],i[0]+i[1]])
                            index += 1
                            interval[index][0] = i[0]+i[1]
                            height.append(max(height[-1],maxheight+i[1]))
                        else:
                            allheight.insert(index,maxheight+i[1])
                            interval.insert(index,[i[0],i[0]+i[1]])
                            height.append(max(height[-1],maxheight+i[1]))
                    else:
                        maxheight = 0
                        while index < len(interval) and (i[0]+i[1]) >= interval[index][1]:
                            maxheight = max(maxheight,allheight[index])
                            allheight.pop(index)
                            interval.pop(index)
                        if index == len(interval):
                            allheight.append(maxheight + i[1])
                            interval.append([i[0],i[0]+i[1]])
                            height.append(max(height[-1],maxheight+i[1]))
                        elif (i[0]+i[1]) > interval[index][0]:
                            maxheight = max(maxheight,allheight[index])
                            allheight.insert(index,maxheight+i[1])
                            interval.insert(index,[i[0],i[0]+i[1]])
                            index += 1
                            interval[index][0] = i[0]+i[1]
                            height.append(max(height[-1],maxheight+i[1]))
                        else:
                            allheight.insert(index,maxheight+i[1])
                            interval.insert(index,[i[0],i[0]+i[1]])
                            height.append(max(height[-1],maxheight+i[1]))
        return height
    a = fallingSquares(0,[[7,2],[1,7],[9,5],[1,8],[3,4]])
    print(a)
'''
'''
leetcode 1368  使网格图至少有一条有效路径的最小代价
import heapq
class Solution:
    def minCost(self, grid: list[list[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        visited = {(0,0)}
        newpoint = [(0,0,0)]
        lenthdict = {}
        lenthdict[(0,0)] = 0
        while newpoint:
            h,i,j = heapq.heappop(newpoint)
            if (i,j) == (m-1,n-1):
                return h
            if grid[i][j] == 1:
                if j < n-1 and ((i,j+1) not in visited or lenthdict[(i,j+1)] > h):
                    heapq.heappush(newpoint,(h,i,j+1))
                    visited.add((i,j+1))
                    lenthdict[(i,j+1)] = h
                if (i,j+1) == (m-1,n-1):
                    return h
            if grid[i][j] == 2:
                if j > 0 and ((i,j-1) not in visited or lenthdict[(i,j-1)] > h):
                    heapq.heappush(newpoint,(h,i,j-1))
                    visited.add((i,j-1))
                    lenthdict[(i,j-1)] = h
            if grid[i][j] == 3:
                if i < m-1 and ((i+1,j) not in visited or lenthdict[(i+1,j)] > h):
                    heapq.heappush(newpoint,(h,i+1,j))
                    visited.add((i+1,j))
                    lenthdict[(i+1,j)] = h
                if (i+1,j) == (m-1,n-1):
                    return h
            else:
                if i > 0 and ((i-1,j) not in visited or lenthdict[(i-1,j)] > h):
                    visited.add((i-1,j))
                    lenthdict[(i-1,j)] = h
                    heapq.heappush(newpoint,(h,i-1,j))

            for x,y in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if 0<=x<m and 0<=y<n and ((x,y) not in visited or lenthdict[(x,y)] > h+1):
                    visited.add((x,y))
                    lenthdict[(x,y)] = h+1
                    heapq.heappush(newpoint,(h+1,x,y))
    a = minCost(0,[[1,1,3],[3,2,2],[1,1,4]])
    print(a)
'''