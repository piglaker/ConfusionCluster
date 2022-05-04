# -*- coding: utf-8 -*-

import Levenshtein
import pypinyin

class calcuate():
    def __init__(self):

        # 字典初始化
        self.bihuashuDict = self.initDict('./db/bihuashu_2w.txt')
        self.hanzijiegouDict = self.initDict('./db/hanzijiegou_2w.txt')
        self.pianpangbushouDict = self.initDict('./db/pianpangbushou_2w.txt')
        self.sijiaobianmaDict = self.initDict('./db/sijiaobianma_2w.txt')

        # 权重定义（可自行调整）
        self.hanzijiegouRate = 10
        self.sijiaobianmaRate = 8
        self.pianpangbushouRate = 6
        self.bihuashuRate = 2
        self.pinyinRate = 10

        return
    
    def initDict(self,path):
       dict = {}
       with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                # 移除换行符，并且根据空格拆分
                splits = line.strip('\n').split(' ')
                key = splits[0]
                value = splits[1]
                dict[key] = value
       return dict

    # 计算核心方法
    '''
    desc: 笔画数相似度
    '''
    def bihuashuSimilar(self,charOne, charTwo):
        
        valueOne, valueTwo = None, None
        if charOne in self.bihuashuDict:
            valueOne = self.bihuashuDict[charOne]
        if charTwo in self.bihuashuDict:
            valueTwo = self.bihuashuDict[charTwo]

        if not valueOne or not valueTwo:
            return 0

        numOne = int(valueOne)
        numTwo = int(valueTwo)

        diffVal = 1 - abs((numOne - numTwo) / max(numOne, numTwo))
        return self.bihuashuRate * diffVal * 1.0


    '''
    desc: 汉字结构数相似度
    '''
    def hanzijiegouSimilar(self,charOne, charTwo):
        valueOne, valueTwo = None, None
        if charOne in self.hanzijiegouDict:
            valueOne = self.hanzijiegouDict[charOne]
        if charTwo in self.hanzijiegouDict:
            valueTwo = self.hanzijiegouDict[charTwo]
        if not valueOne or not valueTwo:
            return 0
        
        if valueOne == valueTwo:
            # 后续可以优化为相近的结构
            return self.hanzijiegouRate * 1
        return 0

    '''
    desc: 四角编码相似度
    '''
    def sijiaobianmaSimilar(self,charOne, charTwo):
        valueOne, valueTwo = None, None
        if charOne in self.sijiaobianmaDict:
            valueOne = self.sijiaobianmaDict[charOne]
        if charTwo in self.sijiaobianmaDict:
            valueTwo = self.sijiaobianmaDict[charTwo]

        if not valueOne or not valueTwo:
            return 0

        totalScore = 0.0
        minLen = min(len(valueOne), len(valueTwo))

        for i in range(minLen):
            if valueOne[i] == valueTwo[i]:
                totalScore += 1.0

        totalScore = totalScore / minLen * 1.0
        return totalScore * self.sijiaobianmaRate

    '''
    desc: 偏旁部首相似度
    '''
    def pianpangbushoutSimilar(self,charOne, charTwo):
        
        valueOne, valueTwo = None, None
        if charOne in self.pianpangbushouDict:
            valueOne = self.pianpangbushouDict[charOne]
        if charTwo in self.pianpangbushouDict:
            valueTwo = self.pianpangbushouDict[charTwo]

        if not valueOne or not valueTwo:
            return 0

        if valueOne == valueTwo:
            # 后续可以优化为字的拆分
            return self.pianpangbushouRate * 1
        return 0
    '''
    desc: 拼音相似度
    '''
    def pinyinSimilar(self,charOne,charTwo):
        pinyinOne = pypinyin.pinyin(charOne)[0][0]
        pintinTwo = pypinyin.pinyin(charTwo)[0][0]

        total_score = Levenshtein.ratio(pinyinOne, pintinTwo)

        return self.pinyinRate * total_score
    '''
    desc: 计算两个汉字的相似度
    '''
    def similar(self,charOne, charTwo):
        if charOne == charTwo:
            return 1.0

        sijiaoScore = self.sijiaobianmaSimilar(charOne, charTwo)
        jiegouScore = self.hanzijiegouSimilar(charOne, charTwo)
        bushouScore = self.pianpangbushoutSimilar(charOne, charTwo)
        bihuashuScore = self.bihuashuSimilar(charOne, charTwo)
        pinyinScore = self.pinyinSimilar(charOne,charTwo)

        totalScore = sijiaoScore + jiegouScore + bushouScore + bihuashuScore + pinyinScore
        totalRate = self.hanzijiegouRate + self.sijiaobianmaRate + self.pianpangbushouRate + self.bihuashuRate + self.pinyinRate

        result = totalScore*1.0 / totalRate * 1.0
        #print('总分：' + str(totalScore) + ', 总权重: ' + str(totalRate) +', 结果:' + str(result))
        #print('四角编码：' + str(sijiaoScore))
        #print('汉字结构：' + str(jiegouScore))
        #print('偏旁部首：' + str(bushouScore))
        #print('笔画数：' + str(bihuashuScore))
        #print('拼音 ' + str(pinyinScore))
        return result


if __name__ == "__main__":
    calcuate_score = calcuate()
    calcuate_score.similar('末', '未')

