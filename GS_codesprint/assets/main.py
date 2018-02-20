#!/bin/python3

#!/bin/python3

import os
import sys
import re
import copy

hash = {}

class Current:
    def __init__(self,order_id,timestamp,symbol,order_type,side,price,quantity):
        self.order_id = order_id
        self.timestamp = timestamp
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.price = price
        self.quantity = quantity
        self.ismatched = 0
        
class OrderList:
    orderBook = {} 
    time = 0

def Type(string):
    patInt = re.search('\d+',string)
    patFloat = re.search('\d+\.\d+', string)
    patStr = re.search('\S+', string)
    if(patFloat):
        return float
    elif(patInt):
        return int
    elif(patStr):
        return str

def splitN(q):
    #  A,2,0000001,XYZ,L,B,103.53,150
    return [int(q.split(',')[1]),int(q.split(',')[2]),(q.split(',')[3]),(q.split(',')[4]),(q.split(',')[5]),float(q.split(',')[6]),int(q.split(',')[7])]

def all_delete():
    for i in list(OrderList.orderBook):
        if(OrderList.orderBook[i].quantity<=0):
            del OrderList.orderBook[i]

def amend_order(q):
    s = ''
    temp = True
    #  A,2,0000001,XYZ,L,B,103.53,150
    orderId = int(q.split(',')[1])

    time = int(q.split(',')[2])

    symbol = (q.split(',')[3])

    order_type = (q.split(',')[4])

    side = (q.split(',')[5])

    price = float(q.split(',')[6])

    quantity = int(q.split(',')[7])

    if(orderId not in OrderList.orderBook):
        return str(orderId)+" - "+"AmendReject - 404 - Order does not exist"

    order = OrderList.orderBook[orderId]
    sp = splitN(q)
    if(order.symbol!= symbol or
     order.side != side or 
     order.order_id!=orderId or 
     order.order_type!=order_type):
        temp = False

    if(check_type(q) and orderId in OrderList.orderBook and 
        len(q.split(','))==8 and time >= OrderList.time and temp):
        if order.ismatched >= quantity:
            order.quantity = 0
        else:
            order.quantity = quantity - order.ismatched
        order.price = price
        #order.timestamp = time OrderList.latestTime = time
        OrderList.orderBook[orderId] = order 
        s = str(orderId)+ " - " + "AmendAccept"

    else:
        s = str(orderId) + ' - ' + "AmendReject - 101 - Invalid amendment details"
    return s

def match_order(query):
    lst = query.split(",")
    if(len(lst)==2):
        return match(int(lst[1]),None)
    else:
        return match(int(lst[1]),lst[2])  

def match(timestamp,name_of_symbol):
    matching_order = {}
    matchings = []

    for order_id in list(OrderList.orderBook):
        order =copy.deepcopy(OrderList.orderBook[order_id])
        if order.timestamp<=timestamp:
            symbol = order.symbol
            if(name_of_symbol !=None and symbol != name_of_symbol):
                continue
            matching_order[symbol] = matching_order.get(symbol,[[],[]])
            if(order.side == "B"):
                matching_order[symbol][0].append(order)
            else:
                matching_order[symbol][1].append(order)

    s_symbols = list(matching_order)
    s_symbols.sort()

    if(s_symbols==None or len(s_symbols)==0):
        return None

    for symbol in s_symbols:
        trade_buy = matching_order[symbol][0]
        trade_sell = matching_order[symbol][1]

        trade_buy = list(sorted(trade_buy,key = lambda x: (x.price,-1*x.timestamp)))
        trade_sell = list(sorted(trade_sell,key=lambda x: (-1*x.price,-1*x.timestamp)))

        while(len(trade_buy)>0 and len(trade_sell)>0):

            trade_l = symbol + "|"

            buyt = trade_buy[-1].order_type

            sellt = trade_buy[-1].order_type
            buyID = trade_buy[-1].order_id
            si = trade_sell[-1].order_id
            if(trade_buy[-1].price>=trade_sell[-1].price or trade_buy[-1].order_type == "M" or trade_sell[-1].order_type == "M"):
                if(trade_sell[-1].order_type=="M"):
                    price = trade_buy[-1].price
                else:
                    price = trade_sell[-1].price 
                if(trade_buy[-1].quantity>trade_sell[-1].quantity):
                    OrderList.orderBook[si].quantity  = 0
                    OrderList.orderBook[si].matched += trade_sell[-1].quantity
                    OrderList.orderBook[buyID].quantity -= trade_sell[-1].quantity
                    OrderList.orderBook[buyID].matched += trade_sell[-1].quantity
                    trade_buy[-1].quantity -= trade_sell[-1].quantity
                    trade_quantity = trade_sell[-1].quantity
                    trade_sell.pop()

                elif(trade_buy[-1].quantity<trade_sell[-1].quantity):
                    OrderList.orderBook[buyID].quantity = 0 
                    OrderList.orderBook[buyID].matched += trade_buy[-1].quantity
                    OrderList.orderBook[si].quantity -= trade_buy[-1].quantity
                    OrderList.orderBook[si].matched += trade_buy[-1].quantity
                    trade_sell[-1].quantity -= trade_buy[-1].quantity 
                    trade_quantity = trade_buy[-1].quantity
                    trade_buy.pop()

                else:
                    OrderList.orderBook[buyID].quantity = 0
                    OrderList.orderBook[buyID].matched += trade_buy[-1].quantity
                    OrderList.orderBook[si].quantity = 0
                    OrderList.orderBook[si].matched += trade_buy[-1].quantity
                    trade_quantity = trade_buy[-1].quantity
                    trade_buy.pop()
                    trade_sell.pop()
                
                mb = str(buyID)+","+str(buyt)+","+str(trade_quantity)+","+"{0:.2f}".format(price)
                ms = "{0:.2f}".format(price)+","+str(trade_quantity)+","+str(sellt)+","+str(si)
                trade_l = trade_l+mb+"|"+ms
                print("Match",trade_l)
                matchings.append(trade_l)
                
            else:
                break
        
        for i in trade_buy:
            if(i.order_type=="I"):
                OrderList.orderBook[i.order_id].quanity = 0
        
        for i in trade_sell:
            if(i.order_type=="I"):
                OrderList.orderBook[i.order_id].quantity = 0 
    all_delete() 
    return matchings

def check_type(query):
    l = query.split(',')
    if(Type(l[1]) is int and
      Type(l[2]) is int and
      Type(l[3]) is str and l[3].isupper() and
      (l[4]=='L' or l[4]=='M' or l[4]=='I') and
      (l[5]=='S' or l[5]== 'B') and
      Type(l[6]) is float and
      (float(l[6])>0 and int(l[7])>0) and
      Type(l[7]) is int):
        return True
    return False

def cancel_order(query):
    orderId = int(query.split(",")[1])
    if(orderId in list(OrderList.orderBook)):
        del OrderList.orderBook[orderId]
        return str(orderId) +  " - " +"CancelAccept" 
    return str(orderId)+ " - " +"CancelReject - 404 - Order does not exist"

def new_order(q):
    s = ''
    orderId = int(q.split(',')[1])
    time = int(q.split(',')[2])
    if(check_type(q) and not orderId in OrderList.orderBook and len(q.split(','))==8 and time >= OrderList.time):
        s = str(orderId) + ' - ' + "Accept"
        hash[orderId] = splitN(q)
        order = splitN(q)
        OrderList.orderBook[orderId] = Current(order[0],order[1],order[2],order[3],order[4],order[5],order[6])
        OrderList.time = time
    else:
        s = str(orderId) + ' - ' + "Reject - 303 - Invalid order details"
    return s

    
def processQueries(queries):
    lst = []
    for q in queries:
        check = q.split(',')[0]
        if(check=='N'):
            lst.append(new_order(q))

        if(check=='X'):
            lst.append(cancel_order(q))

        if(check=='A'):
            lst = lst + amend_order(q)

        if(check=='M'):
            lst.append(match_order(q))

        if(check=='Q'):
            lst.append(query_order(q)) 
    return lst
            

if __name__ == '__main__':
    f = open(os.environ['OUTPUT_PATH'], 'w')

    queries_size = int(input())

    queries = []
    for _ in range(queries_size):
        queries_item = input()
        queries.append(queries_item)

    response = processQueries(queries)
    f.write("\n".join(response))

    f.write('\n')

    f.close()