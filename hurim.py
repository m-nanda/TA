'''
'''

import queue
from functools import cmp_to_key
from collections import OrderedDict
import numpy as np	

class Node:
  '''
  Class untuk Node
  '''
  name = ""	# Node's Item Name
  count = None	# Node's Support Count
  nu = None	# Node's Overestimated Utility
  parent = None	# Node's Parent
  hlink = None	# Points to node whose item name is same as N.name
  children = None	# Children node's of current node
  level = None
  mnu = None

  def __init__(self,name,parent = None,nu=0,mnu=999999999):
    self.name = name
    self.count = 1
    self.nu = nu
    self.parent = parent
    self.children = {}
    self.mnu = mnu
    if self.parent!=None:
      self.level = self.parent.level + 1
    else:
      self.level = 0

  def show(self, level=0, show=False):
    '''
    Fungsi untuk menunjukkan nama node, nu, count-nya, dan mnu
    '''
    if (self.parent != None and show==True):
      print('.', end=' ')
      #print('([^{},{}], -> [{}({},{}), {}]). level: {}'.format(str(self.parent.name),
                                   #str(self.parent.nu),
                                   #self.name, self.nu, self.count, self.mnu,
                                   #level))
    if (self.parent != None and show==False):
      print('.', end=' ')
      #print('([^{},{}], -> [{}({},{}), {}]).'.format(str(self.parent.name),
                               #str(self.parent.nu),
                               #self.name, self.nu, self.count, self.mnu,))

  def insert_child_node(self,i,val,mnu):
    '''
    Fungsi untuk menambahkan/update node anak
    '''

    # jika node i, sudah ada update nu, count, dan mnu nya
    if i in list(self.children.keys()):
      node = self.children[i]
      node.count += 1
      node.nu += val
      node.mnu_awal = node.mnu
      node.mnu = min(node.mnu,mnu)

    # jika node i belum ada, masukkan ke dalam tree
    else:
      self.children[i] = Node(i,self,val,mnu)
    return self.children[i]


class HeaderTable:
  '''
  Class untuk header table
  '''
  table = None

  def __init__(self,items):
    self.table = {item : {"utility":0,"link":None,"last":None} for item in items}

  def show(self):
    '''
    Fungsi untuk menampilkan header tabel
    '''
    print('\n  {:^54s}'.format('Header Table'))
    print('='*58)
    print('| {:^30s} | {:^14s} | {:^5s}|'.format("name","utility","link"))
    print('='*58)
    for item in self.table.keys():
      print('| {:30s} | {:14.2f} | {} |'.format(item, round(self.table[item]["utility"], 2), self.table[item]["link"]))
    print('='*58, end='\n\n')

  def increment_utility(self,item_name,increment):
    '''
    Fungsi untuk menambah utilitas item pada header table

    Input:
      - Item_name: nama item
      - increment: jumlah penambahan
    '''
    if item_name in list(self.table.keys()): 
      self.table[item_name]["utility"] += increment
      return True
    else:
      return False

  def dgu(self,min_util):
    '''
    Fungsi untuk mengambil kandidat item dengan kriteria utilitas yang sesuai
    '''
    self.table = {k: v for k, v in self.table.items() if v["utility"] >= min_util}

  def dlu(self,min_util):
    '''
    Fungsi untuk mengambil kandidat item dengan kriteria utilitas yang sesuai
    '''
    self.table = {k: v for k, v in self.table.items() if v["utility"] >= min_util}

  def dpred(self, tes_item):
    # print(tes_item)
    # print({k: v for k, v in self.table.items() if k in tes_item min_util})
    self.table = {k: v for k, v in self.table.items() if k in tes_item}
    # {key: self.table[key] for key in ky_lit}

        
class UPTree:
  '''
  Class untuk UPTree
  '''
  item_set 				= None
  header_table 			= None
  tree_root				= None
  profit_hash 			= None
  min_util				= None
  max_sup 				= None #int(1*len(database_file))
  current_pattern_base	= ""
  infinity 				= 9999999
  database_file			= None
  profit_table			= None
  test_item = None

  def __init__(self,db=None,profit_hash=None,min_util=None, max_sup=None, tesItem=None):
    self.profit_hash 			= profit_hash
    if profit_hash != None:
      self.item_set 				= list(profit_hash.keys())
      self.header_table 			= HeaderTable(self.item_set)
    if min_util == None:
      self.min_util = 0
    self.min_util				= min_util
    self.max_sup				= max_sup#int(max_sup*len(database_file))
    self.tree_root				= Node("Root")
    self.database_file			= db
    self.profit_table			= profit_hash
    self.test_item = tesItem
    # print('create!')
    # print(tesItem, self.test_item)


  def from_patterns(self,pattern_base,min_util,x):
    self.current_pattern_base = x
    item_set = []
    for patterns in pattern_base:
      [pattern,support,cost] = patterns
      for [item,mnu] in pattern:
        if item not in item_set:
          item_set.append(item)
    self.item_set = item_set
    self.header_table = HeaderTable(self.item_set)
    self.min_util = min_util
    for patterns in pattern_base:
      [pattern,support,cost] = patterns
      for [item,mnu] in pattern:
        self.header_table.increment_utility(item,cost*support)
    self.header_table.table = dict(OrderedDict(sorted(self.header_table.table.items(), key=lambda x: x[1]['utility'], reverse=True)))
    self.header_table.dlu(self.min_util)
    for i in range(len(pattern_base)):
      [pattern,support,cost] = pattern_base[i]
      new_pattern = []
      for [item,mnu] in pattern:
        present = bool(item in list(self.header_table.table.keys()))
        if present:
          new_pattern.append([item,mnu])
        if not present:
          pattern_base[i][2] -= (mnu*support) 
      pattern_base[i][0] = new_pattern
    for i in range(len(pattern_base)):
      pattern_base[i][0] = sorted(pattern_base[i][0], key=cmp_to_key(lambda x,y: self.get_head_val(y) - self.get_head_val(x)))
    for patterns in pattern_base:
      [pattern,support,cost] = patterns
      if len(pattern)==0:
        continue
      current_node = self.tree_root
      sum_mnu_coming_after = 0 
      for [i,mnu] in pattern[1:]:
        sum_mnu_coming_after += mnu*support 
      current_val = cost - sum_mnu_coming_after 
      current_node = current_node.insert_child_node(pattern[0][0],current_val,pattern[0][1]) 
      for [item,mnu] in pattern[1:]:
        current_val += mnu*support 
        current_node = current_node.insert_child_node(item,current_val,mnu) 

  def get_head_val(self,item_mnu):
    [item,mnu] = item_mnu
    return self.header_table.table[item]["utility"]

  def calculate_tu(self,row): 
    Transaction_Utility = 0
    for item in row:
      item_name = item[0]
      quantity  = item[1]
      item_value = self.profit_table[item_name]*quantity
      Transaction_Utility += item_value
    return Transaction_Utility

  def insert_reorganized_transaction(self,transaction):
    current_node = self.tree_root
    current_val = 0
    for i in transaction:
      item = i[0]
      quantity = i[1]
      nu = self.profit_hash[item]*quantity
      current_val += nu
      current_node = current_node.insert_child_node(item,current_val,nu)

  def show_header_table(self):
    self.header_table.show()

  def dbscan_df(self):
    for u in range(self.database_file.shape[0]):
      tu = self.calculate_tu(self.database_file.iloc[u]['Gejala'])
      for item in self.database_file.iloc[u]['Gejala']:
        self.header_table.increment_utility(item[0],tu)
    self.header_table.table = {k: v for k, v in self.header_table.table.items() if v["utility"] > 0}    
    self.header_table.table = dict(OrderedDict(sorted(self.header_table.table.items(), key=lambda x: x[1]['utility'], reverse=True)))
    if len(list(self.header_table.table.keys())) > 0:
      self.min_util = self.min_util * self.header_table.table[list(self.header_table.table.keys())[0]]['utility']

  def reorganized_dbscan_dgn_df(self,show=False):
    #print("Showing Reorganized DB and Applying DGN")
    for u in range(self.database_file.shape[0]):
      filtered_row = []
      for item in self.database_file.iloc[u]['Gejala']:
        if item[0] in list(self.header_table.table.keys()):
          filtered_row.append(item)        
      self.database_file.iloc[u]['Gejala'] = sorted(filtered_row, 
                                                     key=cmp_to_key(lambda x,y: self.header_table.table[y[0]]["utility"] - self.header_table.table[x[0]]["utility"]))
      tu = self.calculate_tu(self.database_file.iloc[u]['Gejala'])
      self.insert_reorganized_transaction(sorted(filtered_row,
                                                       key=cmp_to_key(lambda x,y: self.header_table.table[y[0]]["utility"] - self.header_table.table[x[0]]["utility"])))

      if(show):
        if self.database_file.shape[0] < 10:
          print('{}. {:180s} | (TU: {})'.format(u+1, str(self.database_file.iloc[u]['Gejala']), tu))
        else:
          if u<3:
            print('{}. {:180s} | (TU: {})'.format(u+1, str(self.database_file.iloc[u]['Gejala']), tu))
          elif u in range(self.database_file.shape[0]-3,self.database_file.shape[0]):
            print('{}. {:180s} | (TU: {})'.format(u+1, str(self.database_file.iloc[u]['Gejala']), tu))
          elif u in np.linspace(3, self.database_file.shape[0]-2, 10, dtype=int):
            print('......')

  def dgu(self):
    self.header_table.dgu(self.min_util)

  def d_pred(self):
    # print(self.test_item)
    self.header_table.dpred(self.test_item)

  def show_tree(self):
    q = queue.Queue()
    current_level = 0
    q.put(self.tree_root)
    member_lvl_count={}
    while not q.empty():
      n = q.get()
      if(n.level!=current_level):
        current_level=n.level
      if current_level not in member_lvl_count:
        member_lvl_count[current_level]=1
      else:
        member_lvl_count[current_level]+=1
      if(n.name!="Root"):
        if(n.name not in self.header_table.table.keys()):
          continue
        elif (self.header_table.table[n.name]["link"]==None):
          self.header_table.table[n.name]["link"] = n
          self.header_table.table[n.name]["last"] = n
        else:
          self.header_table.table[n.name]["last"].hlink = n
          self.header_table.table[n.name]["last"] = n
      for child_node_name in list(n.children.keys()):
        q.put(n.children[child_node_name])

  def hurim_upraregrowth(self):
    phui = []
    urutan = list(self.header_table.table.keys())
    urutan.reverse()
    for item in urutan: 
      if(self.header_table.table[item]["utility"]>self.min_util):
        item_potential_value = 0
        cpb = []
        sup_list = []
        current = self.header_table.table[item]["link"]
        if(current != None):
          sup=0
          while(True):
            item_potential_value += current.nu
            pb =[[],0,0] # [ [ [pattern,mnu],..],support,cost]
            pb[1] = current.count
            pb[2] = current.nu
            sup_list.append(current.count)
            up = current.parent
            while(up.parent!=None):
              pb[0].append([up.name,up.mnu])
              up = up.parent                
            sup += pb[1]
            if len(pb[0])!=0:
              cpb.append(pb)
            if(current.hlink == None):
              break
            current = current.hlink
          if(item_potential_value>self.min_util and (0 < sup < self.max_sup)):
            phui.append([item,item_potential_value, sup]) #kandidat = phui
        tree = UPTree(min_util=self.min_util, max_sup=self.max_sup)
        tree.from_patterns(cpb,self.min_util,self.current_pattern_base+item)
        tree.show_tree()
        if all([t > self.max_sup for (t) in sup_list]):
          continue
        else:
          retreived = tree.hurim_upraregrowth()
          for i in retreived:
            phui.append([item+', '+i[0],i[1],i[2]])
    return phui

  def solve_df(self):
#     print('. ', end=' ')		
    self.dbscan_df()
    #print('\n', 2, end=': ')
    # self.show_header_table()
#     print('. ', end=' ')		
    self.dgu()
    #print('\n', 4, end=': ')
    #self.show_header_table()
    print('. ', end=' ')		
    self.reorganized_dbscan_dgn_df()
    #print(6, end=':\n')
    self.show_tree()
#     print('. ', end=' ')		
    return self.hurim_upraregrowth()

  def pred(self): #, test_item):
#     print('. ', end=' ')
    self.dbscan_df()
    # self.show_header_table()
#     print('. ', end=' ')
    self.dgu()
    self.d_pred()
    # print('Kandidat Item Berdasarkan Record Data Tes')
    # self.show_header_table()
    self.reorganized_dbscan_dgn_df()
    self.show_tree()
#     print('. ', end=' ')		
    return self.hurim_upraregrowth()
