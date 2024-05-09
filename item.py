import os
import sys
import numpy as np
from dataloader import MS_loader

'''
3029C003AA
3029C004AA
3029C005AA
3029C006AA
3029C009AA
3029C010AA
3030C001AA
3030C002AA
3030C003AA
3030C004AA
3031C001AA
3031C002AA
3031C003AA
'''

# item = ['C550', 'C551', '1525', '3102', 'C160', '8588', '2899', 'V243', 
#         'V242', 'B729', '3698', '3699', '2917', '4357', '3799', '3780',
#         'C254', '0361', '0363', '1506', '1507', '0368', '1503', '1504', 
#         '6415', '6416', '6418', '6420', '6421', '6422', '6423', '6467', 
#         '6323', 'B793', '7630', '0000']

item = 'C550'
MS_list = [['3029C005AA', 'step_1'], ['3029C006AA', 'step_1'], ['3029C009AA', 'step_1'], ['3029C010AA', 'step_1'], 
           ['3030C002AA', 'step_1'], ['3030C003AA', 'step_1'], ['3030C004AA', 'step_1'], ['3031C001AA', 'step_1'], 
           ['3031C002AA', 'step_1'], ['3031C003AA', 'step_1']]

item = 'C551'
MS_list = [['3029C003AA', 'step_1'], ['3029C004AA', 'step_1'], ['3030C001AA', 'step_1']]

item = '1525'
MS_list = [['3029C003AA', 'step_5'], ['3029C004AA', 'step_5'], ['3029C005AA', 'step_5'], ['3029C006AA', 'step_5'], 
           ['3030C001AA', 'step_5'],['3029C009AA', 'step_5'], ['3029C010AA', 'step_5'], ['3030C002AA', 'step_5'], 
           ['3030C003AA', 'step_5'], ['3030C004AA', 'step_5'], ['3031C001AA', 'step_5'], ['3031C002AA', 'step_5'], 
           ['3031C003AA', 'step_5'],
           ['3029C003AA', 'step_7'], ['3029C004AA', 'step_7'], ['3029C005AA', 'step_7'], ['3029C006AA', 'step_7'], 
           ['3030C001AA', 'step_7'], ['3029C009AA', 'step_7'], ['3029C010AA', 'step_7'], ['3030C002AA', 'step_7'], 
           ['3030C003AA', 'step_7'], ['3030C004AA', 'step_7'], ['3031C001AA', 'step_7'], ['3031C002AA', 'step_7'], 
           ['3031C003AA', 'step_7']]

item = '3102'
MS_list = [['3029C003AA', 'step_4'], ['3029C004AA', 'step_4'], ['3029C005AA', 'step_4'], ['3029C006AA', 'step_4'], 
           ['3030C001AA', 'step_4'], ['3029C009AA', 'step_4'], ['3029C010AA', 'step_4'], ['3030C002AA', 'step_4'], 
           ['3030C003AA', 'step_4'], ['3030C004AA', 'step_4'], ['3031C001AA', 'step_4'], ['3031C002AA', 'step_4'], 
           ['3031C003AA', 'step_4']]

item = 'C160'
MS_list = [['3029C004AA', 'step_13'], ['3029C010AA', 'step_15']]

item = '8588'
MS_list = [['3029C010AA', 'step_8']]

item = '2899'
MS_list = [['3029C010AA', 'step_9']]

item = 'V243'
MS_list = [['3029C004AA', 'step_10'], ['3029C004AA', 'step_11']]

item = 'V242'
MS_list = [['3029C003AA', 'step_10'], ['3029C005AA', 'step_10'], ['3029C006AA', 'step_10'], ['3029C009AA', 'step_10'],
           ['3029C010AA', 'step_12'], ['3031C001AA', 'step_10'], ['3031C002AA', 'step_10'], ['3031C003AA', 'step_10'], 
           ['3029C003AA', 'step_11'], ['3029C005AA', 'step_11'], ['3029C006AA', 'step_11'], ['3029C009AA', 'step_11'],
           ['3029C010AA', 'step_13'], ['3031C001AA', 'step_11'], ['3031C002AA', 'step_11'], ['3031C003AA', 'step_11']]

item = 'B792'
MS_list = [['3030C001AA', 'step_10'], ['3030C002AA', 'step_10'], ['3030C003AA', 'step_10'], ['3030C004AA', 'step_10'], 
           ['3030C001AA', 'step_11'], ['3030C002AA', 'step_11'], ['3030C003AA', 'step_11'], ['3030C004AA', 'step_11']]

item = '3698'
MS_list = [['3029C003AA', 'step_13'], ['3029C004AA', 'step_16'], ['3029C005AA', 'step_12'], ['3029C006AA', 'step_12'],
           ['3029C009AA', 'step_13'], ['3029C010AA', 'step_18'], ['3030C001AA', 'step_13'], ['3030C002AA', 'step_13'],
           ['3030C003AA', 'step_13'], ['3030C004AA', 'step_13'], ['3031C001AA', 'step_13'], ['3031C002AA', 'step_12'],
           ['3031C003AA', 'step_12']]

item = '3699'
MS_list = [['3029C003AA', 'step_4'], ['3029C004AA', 'step_4'], ['3029C005AA', 'step_4'], ['3029C006AA', 'step_4'], 
           ['3030C001AA', 'step_4'], ['3029C009AA', 'step_4'], ['3029C010AA', 'step_4'], ['3030C002AA', 'step_4'], 
           ['3030C003AA', 'step_4'], ['3030C004AA', 'step_4'], ['3031C001AA', 'step_4'], ['3031C002AA', 'step_4'], 
           ['3031C003AA', 'step_4']]

item = '2917'
MS_list = [['3029C003AA', 'step_6'], ['3029C004AA', 'step_6'], ['3029C005AA', 'step_6'], ['3029C006AA', 'step_6'], 
           ['3030C001AA', 'step_6'], ['3029C009AA', 'step_6'], ['3029C010AA', 'step_6'], ['3030C002AA', 'step_6'], 
           ['3030C003AA', 'step_6'], ['3030C004AA', 'step_6'], ['3031C001AA', 'step_6'], ['3031C002AA', 'step_6'], 
           ['3031C003AA', 'step_6']]

item = '4357'
MS_list = [['3029C010AA', 'step_9']]

item = '3779'
MS_list = [['3029C003AA', 'step_12'], ['3029C009AA', 'step_12'], ['3030C001AA', 'step_12'], ['3030C002AA', 'step_12'],
           ['3030C003AA', 'step_12'], ['3030C004AA', 'step_12'], ['3031C001AA', 'step_12'], ['3029C003AA', 'step_14'],
           ['3029C009AA', 'step_14'], ['3030C001AA', 'step_14'], ['3030C002AA', 'step_14'], ['3030C003AA', 'step_14'], 
           ['3030C004AA', 'step_14'], ['3031C001AA', 'step_14']]

item = '3780'
MS_list = [['3029C003AA', 'step_9'], ['3029C004AA', 'step_9'], ['3029C005AA', 'step_9'], ['3029C006AA', 'step_9'], 
           ['3030C001AA', 'step_9'], ['3029C009AA', 'step_9'], ['3029C010AA', 'step_11'], ['3030C002AA', 'step_9'], 
           ['3030C003AA', 'step_9'], ['3030C004AA', 'step_9'], ['3031C001AA', 'step_9'], ['3031C002AA', 'step_9'], 
           ['3031C003AA', 'step_9']]

item = 'C254'
MS_list = [['3029C004AA', 'step_14'], ['3029C010AA', 'step_16']]

item = '0361'
MS_list = [['3029C004AA', 'step_12'], ['3029C010AA', 'step_14']]

item = '0363'
MS_list = [['3029C004AA', 'step_17'], ['3029C010AA', 'step_19']]

item = '1506'
MS_list = [['3029C010AA', 'step_19']]

item = '1507'
MS_list = [['3029C010AA', 'step_17']]

item = '0368'
MS_list = [['3029C004AA', 'step_15'], ['3029C010AA', 'step_17']]

item = '1503'
MS_list = [['3029C010AA', 'step_17']]

item = '1504'
MS_list = [['3029C004AA', 'step_15']]

item = '6415'
MS_list = [['3031C001AA', 'step_3'], ['3031C002AA', 'step_3'], ['3031C003AA', 'step_3']]

item = '6416'
MS_list = [['3029C003AA', 'step_3'], ['3029C005AA', 'step_3'], ['3029C006AA', 'step_3'], ['3029C009AA', 'step_3']]

item = '6420'
MS_list = [['3030C001AA', 'step_3'], ['3030C002AA', 'step_3'], ['3030C003AA', 'step_3']]

item = '6421'
MS_list = [['3030C004AA', 'step_3']]

item = '6423'
MS_list = [['3029C010AA', 'step_3']]

item = '6467'
MS_list = [['3029C004AA', 'step_3']]

item = '6323'
MS_list = [['3029C003AA', 'step_3'], ['3029C004AA', 'step_3'], ['3029C005AA', 'step_3'], ['3029C006AA', 'step_3'], 
           ['3030C001AA', 'step_3'], ['3029C009AA', 'step_3'], ['3029C010AA', 'step_3'], ['3030C002AA', 'step_3'], 
           ['3030C003AA', 'step_3'], ['3030C004AA', 'step_3'], ['3031C001AA', 'step_3'], ['3031C002AA', 'step_3'], 
           ['3031C003AA', 'step_3']]

item = 'B793'
MS_list = [['3029C003AA', 'step_2'], ['3029C004AA', 'step_2'], ['3029C005AA', 'step_2'], ['3029C006AA', 'step_2'], 
           ['3030C001AA', 'step_2'], ['3029C009AA', 'step_2'], ['3029C010AA', 'step_2'], ['3030C002AA', 'step_2'], 
           ['3030C003AA', 'step_2'], ['3030C004AA', 'step_2'], ['3031C001AA', 'step_2'], ['3031C002AA', 'step_2'], 
           ['3031C003AA', 'step_2']]

item = '0000'
MS_list = [['3029C003AA', 'step_8'], ['3029C004AA', 'step_8'], ['3029C005AA', 'step_8'], ['3029C006AA', 'step_8'], 
           ['3030C001AA', 'step_8'], ['3029C009AA', 'step_8'], ['3029C010AA', 'step_10'], ['3030C002AA', 'step_8'], 
           ['3030C003AA', 'step_8'], ['3030C004AA', 'step_8'], ['3031C001AA', 'step_8'], ['3031C002AA', 'step_8'], 
           ['3031C003AA', 'step_8']]