import itertools
import json
import itertools
import numpy as np

#--separableな選好の抽出-----------------------------------------------------------------------------------------------------
def JudgeSeparable(preference):
    type01 = {1: [], 2: [], 3: []}
    type02 = {1: [], 2: [], 3: []}
    for bundle in preference:
        type01[bundle[0]].append(bundle[1])
        type02[bundle[1]].append(bundle[0])
    for key in range(2, 4):
        if type01[1] != type01[key] or type02[1] != type02[key]:
            return False
    return True

product = list(itertools.product([1,2,3], repeat=2))
preferences = itertools.permutations(product)
sep_preference_list=[]

for preference in preferences:
    if JudgeSeparable(preference) == True:
        sep_preference_list.append(preference)

#separableな選好をjsonファイルsep_preference.jsonに保存
with open('sep_preference.json', "a",encoding='utf-8') as file:
    json.dump(sep_preference_list,file,ensure_ascii=False)
    file.write('\n')

#---------------------------------------------------------------------------------------------------------------------------

#--関数μの処理---------------------------------------------------------------------------------------------------------------

#選好を表す配列を引数に与えるとR_im × 5のリストを返す関数
def GenMirrorPreference(preference):
    mirror_preference_list = []
    mirrors = itertools.permutations((1,2,3))

    for mirror in mirrors:
        mirror_preference = []
        for bundle in preference:
            mirror_bundle = []
            for i in bundle:
                if i == 1:
                    mirror_bundle.append(mirror[0])
                if i == 2:
                    mirror_bundle.append(mirror[1])
                if i == 3:
                    mirror_bundle.append(mirror[2])
            mirror_preference.append(mirror_bundle)
        mirror_preference_list.append(mirror_preference)

    return mirror_preference_list

# separableな選好のリストをロード
with open("sep_preference.json",'r') as f:
    sep_preference_list = json.load(f)

# 各選好Rに対して、リスト[Ri,Ri1,...,Ri5]を作る。
# ただし、選好の配列ではなく、sep_preference_listのインデックスを格納する。
mirror_table_list = []
for preference in sep_preference_list:
    mirror_preference_list = GenMirrorPreference(preference)
    mirror_list = []
    for mirror_preference in mirror_preference_list:
        index = sep_preference_list.index(mirror_preference)
        mirror_list.append(index)

    mirror_table_list.append(mirror_list)

# 1512個のリストを要素に持つリストをsep_mirror_table.jsonに保存
with open("sep_mirror_table.json","w")as f:
    json.dump(mirror_table_list,f)


# 選好プロファイル(インデックス)を生成するジェネレータ
def GenProfileIter():
    sep_preference = [i for i in range(0,1512)]
    for element in itertools.product(sep_preference,repeat=3):
        yield element

# 関数μについて、プロファイルR(インデックス)と、mを入れたら、R1m,R2m,R3mを各mに従って入れ替えたものを返す関数
def ReplacementPreference(profile,mirror):
    if mirror == 1:
        profile[0],profile[1],profile[2] = profile[0],profile[2],profile[1]
    elif mirror == 2:
        profile[0],profile[1],profile[2] = profile[1],profile[0],profile[2]
    elif mirror == 3:
        profile[0],profile[1],profile[2] = profile[2],profile[0],profile[1]
    elif mirror == 4:
        profile[0],profile[1],profile[2] = profile[1],profile[2],profile[0]
    elif mirror == 5:
        profile[0],profile[1],profile[2] = profile[2],profile[1],profile[0]
    return profile

with open("sep_mirror_table.json",'r') as f:
    sep_mirror_table = json.load(f)

# 選好プロファイルR(インデックス)を入れたらリスト[μ1(R),...,μ5(R)](インデックス)を返す関数。
def GenMirrorList(profile):
    mirror_list = []

    for mirror in range(1,6):
        mirror_profile = []
        for preference in profile:
            mirror_profile.append(sep_mirror_table[preference][mirror])

        mirror_profile = tuple(ReplacementPreference(mirror_profile,mirror))

        mirror_list.append(mirror_profile)

    return mirror_list

#1512**3のテンソルを作成し、μにより、プロファイルを仕分ける関数
def ExtractOriginal():
    mirror_list = []
    tensor = np.ones((1512, 1512, 1512), dtype=np.uint8)
            
    for profile in GenProfileIter():

        if tensor[profile] == 1:
            mirror_list = GenMirrorList(profile)
            for mirror in mirror_list:
                #特殊ケースのm=3,m=4のための処理。μmとRが一致したらbreak
                if mirror == profile:
                    break
                else:
                    # μ1(R),...,μ5(R)に0を割り当てていく
                    tensor[mirror] = 0
        else:
            pass
    # tensorを.npyファイルに保存
    np.save('tensor_data.npy', tensor)

ExtractOriginal()

#---------------------------------------------------------------------------------------------------------------------------

#--手順1。選好プロファイルの分類-----------------------------------------------------------------------------------------------

def TopTradingRule(sep_profile):
    endowments_profile = {}
    allocation = [None,None,None]
    participant = [1,2,3]

    #--Ⅰ.選好プロファイルを変形--
    i=1
    for preference in sep_profile:
        endowments_profile[i] = []
        for bundle in preference:
            same = False
            for goods in bundle:
                if bundle[0] == goods:
                    same = True
                else:
                    same = False
                    break
            if same == True:
                endowments_profile[i].append(bundle[0])
        i+=1

    #市場参加者がいなくなるまで以下の処理を繰り返す。
    while participant != []: 
        #--II.自己サイクルを処理--
        for agent in participant:
            if agent == endowments_profile[agent][0]: #自己サイクルができた時

                bundle = [agent,agent] #初期保有bundleを作る
                allocation[agent-1] = bundle #最終的な配分に追加

                participant.remove(agent) #市場参加者から取り除く
                
                #プロファイルから自己サイクルの個人と財を除外
                del endowments_profile[agent] 
                for order in endowments_profile:
                    if agent in endowments_profile[order]:
                        endowments_profile[order].remove(agent)

        if participant == []: #この時点で市場参加者が0になれば配分が決まり、処理を終える。
            break

        #--Ⅲ.サイクルを探す--
        best_dict = {} #市場参加者の現時点において最善の財を示す辞書型を作成する。
        for agent in participant:
            best_dict[agent] = endowments_profile[agent][0]

        cycle = []
        cycle.append(best_dict[participant[0]])
        while True:
            cycle.append(best_dict[cycle[-1]])
            if cycle.count(cycle[-1]) == 2: #サイクルができれば以下の処理を行う

                cycle_agent = cycle[cycle.index(cycle[-1]):-1] #サイクルに含まれる個人のリストを作成

                #サイクル内の個人は現時点で最も好きな財を得る。
                for agent in cycle_agent: 
                    bundle = [best_dict[agent],best_dict[agent]]
                    allocation[agent-1] = bundle

                participant = list(set(participant)-set(cycle_agent)) #市場参加者から除外

                for order in endowments_profile: #プロファイルからサイクル内の個人と財を除外
                    for agent in cycle_agent:
                        if agent in endowments_profile[order]:
                            endowments_profile[order].remove(agent)
                break
    return allocation

#バンドルの組み合わせを作るジェネレータ
def GenProduct(*iterrables):
    for element in itertools.product(*iterrables):
        yield element

def GetPdAllocation(sep_profile):
    pd_list = []
    pd_flag = False #パレート支配する配分が存在するかどうかを判定するフラグ。

    #--I.初期保有TTCルールの配分より好ましいバンドルの抽出--
    ttc_allocation = TopTradingRule(sep_profile)
    prefer_bundles = {1: [], 2: [], 3: []}
    for agent in range(1,4):
        prefer_bundles[agent] += list(sep_profile[agent-1][:sep_profile[agent-1].index(ttc_allocation[agent-1])+1])

    if [] in prefer_bundles.values():
        return False
    
    #--II.抽出したバンドルから交換可能な組み合わせを抽出する--

    #初期保有TTCルールの配分より好ましい財の中で、バンドルの組み合わせを作る。

    args = list(prefer_bundles.values())
    combi_iter = GenProduct(*args)

    #各組み合わせの交換可能性判定
    for combi in combi_iter:
        combi_check_dict = {1: [], 2: []}

        for bundle in combi:
            combi_check_dict[1].append(bundle[0])
            combi_check_dict[2].append(bundle[1])
        
        if len(set(combi_check_dict[1])) == 3 and len(set(combi_check_dict[2])) == 3:
            possible = True
            pd_allocation = list(combi)
        else:
            continue

        #可能な組み合わせの内、初期保有TTCルールの配分と一致しないものをリストに追加
        if possible == True and pd_allocation != ttc_allocation:
            pd_flag = True
            pd_list.append(pd_allocation)

    #可能な組み合わせがなければFalseを返す。
    if pd_flag == False: 
        return False

    return pd_list

# RNPEとRPEによって、1と2を割り当てていく関数
def ExtractParetoDominate():
    #separableな選考のリストをjsonから取得
    with open("sep_preference.json",'r') as f:
        sep_preference_list = json.load(f)

    array = np.load('tensor_data.npy')

    for x in range(1512):
        for y in range(1512):
            for z in range(1512):
                if array[x,y,z] == 1:
                    sep_profile = (sep_preference_list[x],sep_preference_list[y],sep_preference_list[z])
                    if GetPdAllocation(sep_profile) == False:
                        array[x,y,z] = 2
                    else:
                        pass

    np.save('tensor_data.npy', array)

ExtractParetoDominate()

#---------------------------------------------------------------------------------------------------------------------------

#--手順2,3,4。プロファイルの探索----------------------------------------------------------------------------------------------
#この関数を3回実行すると、4回目以降は1の総数が変わらなくなる。
def SearchProfile01():

    array = np.load('tensor_data.npy')

    #separableな選考のリストをjsonから取得
    with open("sep_preference.json",'r') as f:
        sep_preference_list = json.load(f)

    for x in range(1512):
        for y in range(1512):
            for z in range(1512):

                #インデックス[0,0,0]から順に見ていって、1が割り当てられている要素に対して以下の処理を行う。
                if array[x,y,z] == 1:

                    sp_flag = False
                    #インデックス[x,y,z]から選好プロファイルの配列を取得
                    profile=[sep_preference_list[x],sep_preference_list[y],sep_preference_list[z]]
                    #[x,y,z]のPD配分を取得
                    pd_list = GetPdAllocation(profile)

                    #該当のプロファイルと選好が一人違いのプロファイルの配列をスライス。
                    slice1 = array[:, y, z]  
                    slice2 = array[x, :, z]  
                    slice3 = array[x, y, :] 

                    #以下スライス1の処理
                    slice_index=0
                    #スライスについて繰り返し。値が2のものについて処理
                    for s in slice1:
                        if s == 2:
                            #スライス上のプロファイルをインデックスから選好プロファイルの配列に戻す
                            slice_profile = [sep_preference_list[slice_index],sep_preference_list[y],sep_preference_list[z]]
                            ttc_allocation = TopTradingRule(slice_profile)
                            #スライスのプロファイルにおいてttcよりPD配分のほうが好ましければ、PD配分をpd_listから削除する。
                            # pd_listが空になれば、プロファイルの値を2にして次のプロファイルへ
                            del_list=[]
                            for pd in pd_list:
                                if slice_profile[0].index(pd[0]) < slice_profile[0].index(ttc_allocation[0]) :
                                    del_list.append(pd)
                            for d in del_list:
                                pd_list.remove(d)
                            if pd_list == []:
                                array[x, y, z] = 2
                                sp_flag = True
                                break
                        slice_index+=1
                    #Rが見つかれば次の1プロファイルへ
                    if sp_flag == True:
                        continue
                    
                    #以下スライス2の処理
                    slice_index=0
                    #スライスについて繰り返し。値が2のものについて処理
                    for s in slice2:
                        if s == 2:
                            #スライス上のプロファイルをインデックスから選好プロファイルの配列に戻す
                            slice_profile = [sep_preference_list[x],sep_preference_list[slice_index],sep_preference_list[z]]
                            ttc_allocation = TopTradingRule(slice_profile)
                            #スライスのプロファイルにおいてttcよりPD配分のほうが好ましければ、PD配分をpd_listから削除する。
                            # pd_listが空になれば、プロファイルの値を2にして次のプロファイルへ
                            del_list=[]
                            for pd in pd_list:
                                if slice_profile[1].index(pd[1]) < slice_profile[1].index(ttc_allocation[1]) :
                                    del_list.append(pd)
                            for d in del_list:
                                pd_list.remove(d)
                            if pd_list == []:
                                array[x, y, z] = 2
                                sp_flag = True
                                break
                        slice_index+=1
                    #Rが見つかれば次の1プロファイルへ
                    if sp_flag == True:
                        continue

                    #以下スライス3の処理
                    slice_index=0
                    #スライスについて繰り返し。値が2のものについて処理
                    for s in slice3:
                        if s == 2:
                            #スライス上のプロファイルをインデックスから選好プロファイルの配列に戻す
                            slice_profile = [sep_preference_list[x],sep_preference_list[y],sep_preference_list[slice_index]]
                            ttc_allocation = TopTradingRule(slice_profile)
                            #スライスのプロファイルにおいてttcよりPD配分のほうが好ましければ、PD配分をpd_listから削除する。
                            # pd_listが空になれば、プロファイルの値を2にして次のプロファイルへ
                            del_list=[]
                            for pd in pd_list:
                                if slice_profile[2].index(pd[2]) < slice_profile[2].index(ttc_allocation[2]) :
                                    del_list.append(pd)
                            for d in del_list:
                                pd_list.remove(d)
                            if pd_list == []:
                                array[x, y, z] = 2
                                sp_flag = True
                                break
                        slice_index+=1
                    #Rが見つかれば次の1プロファイルへ
                    if sp_flag == True:
                        continue
                    elif sp_flag == False:
                        pass

    if np.count_nonzero(array == 1) == 0:
        print("Q.E.D")
        np.save('tensor_data.npy', array)
    else:
        one_count = np.count_nonzero(array == 1)
        print(one_count)
        np.save('tensor_data.npy', array)

SearchProfile01()
SearchProfile01()
SearchProfile01()

#---------------------------------------------------------------------------------------------------------------------------

#--1が割り当てられているプロファイルの、μ1~5を3に-------------------------------------------------------------------------------
array = np.load('tensor_data.npy')
one_indices = np.argwhere(array == 1)

for idx in one_indices:
    x, y, z = idx
    mirror_list = GenMirrorList((x,y,z))

    for profile in mirror_list:
        array[profile] = 3

np.save('tensor_data.npy', array)

#---------------------------------------------------------------------------------------------------------------------------

#--1が割り当てられているプロファイルの、μ1~5を3にした後の手順2,3,4---------------------------------------------------------------
#この関数を2回実行すると1の総数が0になる。
def SearchProfile02():

    array = np.load('tensor_data.npy')
    one_indices = np.argwhere(array == 1)

    #separableな選考のリストをjsonから取得
    with open("sep_preference.json",'r') as f:
        sep_preference_list = json.load(f)

    for idx in one_indices:

        x,y,z = idx
        sp_flag = False
        #[x,y,z]のPD配分を取得
        profile=[sep_preference_list[x],sep_preference_list[y],sep_preference_list[z]]
        pd_list = GetPdAllocation(profile)

        #該当のプロファイルと一つ違いのプロファイルの配列をスライス。
        slice1 = array[:, y, z]  
        slice2 = array[x, :, z]  
        slice3 = array[x, y, :] 

        #以下スライス1の処理
        slice_index=0
        #スライスについて繰り返し。値が0のものについて処理
        for s in slice1:
            if s == 0:
                #スライス上のプロファイルをインデックスから選好プロファイルの配列に戻す
                slice_profile = [sep_preference_list[slice_index],sep_preference_list[y],sep_preference_list[z]]
                ttc_allocation = TopTradingRule(slice_profile)
                #スライスのプロファイルにおいてttcよりPD配分のほうが好ましければ、PD配分をpd_listから削除する。
                # pd_listが空になれば、プロファイルの値を0にして次のプロファイルへ
                del_list=[]
                for pd in pd_list:
                    if slice_profile[0].index(pd[0]) < slice_profile[0].index(ttc_allocation[0]) :
                        del_list.append(pd)
                for d in del_list:
                    pd_list.remove(d)
                if pd_list == []:
                    array[x, y, z] = 0

                    #インデックス[x,y,z]と対応するR'のμ1(R')~μ5(R')を0へ
                    mirror_list = GenMirrorList((x,y,z))
                    for profile in mirror_list:
                        array[profile] = 0

                    sp_flag = True
                    break
            slice_index+=1
        #Rが見つかれば次の1プロファイルへ
        if sp_flag == True:
            continue
        
        #以下スライス2の処理
        slice_index=0
        #スライスについて繰り返し。値が0のものについて処理
        for s in slice2:
            if s == 0:
                #スライス上のプロファイルをインデックスから選好プロファイルの配列に戻す
                slice_profile = [sep_preference_list[x],sep_preference_list[slice_index],sep_preference_list[z]]
                ttc_allocation = TopTradingRule(slice_profile)
                #スライスのプロファイルにおいてttcよりPD配分のほうが好ましければ、PD配分をpd_listから削除する。
                # pd_listが空になれば、プロファイルの値を0にして次のプロファイルへ
                del_list=[]
                for pd in pd_list:
                    if slice_profile[1].index(pd[1]) < slice_profile[1].index(ttc_allocation[1]) :
                        del_list.append(pd)
                for d in del_list:
                    pd_list.remove(d)
                if pd_list == []:
                    array[x, y, z] = 0

                    #インデックス[x,y,z]と対応するR'のμ1(R')~μ5(R')を0へ
                    mirror_list = GenMirrorList((x,y,z))
                    for profile in mirror_list:
                        array[profile] = 0

                    sp_flag = True
                    break
            slice_index+=1
        #Rが見つかれば次の1プロファイルへ
        if sp_flag == True:
            continue

        #以下スライス3の処理
        slice_index=0
        #スライスについて繰り返し。値が0のものについて処理
        for s in slice3:
            if s == 0:
                #スライス上のプロファイルをインデックスから選好プロファイルの配列に戻す
                slice_profile = [sep_preference_list[x],sep_preference_list[y],sep_preference_list[slice_index]]
                ttc_allocation = TopTradingRule(slice_profile)
                #スライスのプロファイルにおいてttcよりPD配分のほうが好ましければ、PD配分をpd_listから削除する。
                # pd_listが空になれば、プロファイルの値を0にして次のプロファイルへ
                del_list=[]
                for pd in pd_list:
                    if slice_profile[2].index(pd[2]) < slice_profile[2].index(ttc_allocation[2]) :
                        del_list.append(pd)
                for d in del_list:
                    pd_list.remove(d)
                if pd_list == []:
                    array[x, y, z] = 0

                    #インデックス[x,y,z]と対応するR'のμ1(R')~μ5(R')を0へ
                    mirror_list = GenMirrorList((x,y,z))
                    for profile in mirror_list:
                        array[profile] = 0

                    sp_flag = True
                    break
            slice_index+=1
        #Rが見つかれば次の1プロファイルへ
        if sp_flag == True:
            continue
        elif sp_flag == False:
            pass
    
    if np.count_nonzero(array == 1) == 0:
        print("Q.E.D")
        np.save('tensor_data.npy', array)
    else:
        one_count = np.count_nonzero(array == 1)
        print(one_count)

        # tensorをnpyファイルに保存
        np.save('tensor_data.npy', array)

SearchProfile02()
SearchProfile02()

