import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import random
import os
import pandas as pd 
file_path = 'Artprosumidores_energia_6pEG.xlsx'
file_path = 'Demandaóptima_Comunidad6p.xlsx'
df = pd.read_excel(file_path)
G1 = df['Glim1'].values
G2 = df['Glim2'].values
G3 = df['Glim3'].values
G4 = df['Glim4'].values
G5 = df['Glim5'].values
G6 = df['Glim6'].values
D1 = df['D*1'].values
D2 = df['D*2'].values
D3 = df['D*3'].values
D4 = df['D*4'].values
D5 = df['D*5'].values
D6 = df['D*6'].values
generation = [[G1[t],G2[t],G3[t],G4[t],G5[t],G6[t]] for t in range(24)]
demand = [[D1[t],D2[t],D3[t],D4[t],D5[t],D6[t]] for t in range(24)]

####Parámetros de entrada
###Utilidad
theta = [10,10,10,10,10,10]
lamda = [100,100,100,100,100,100]
etha0 = [1, 1, 1,1,1,1]
Pgs = 1650
Pgb = 50
precioin = [Pgb,Pgb,Pgb,Pgb, Pgb,Pgb] # [Pgs/2, Pgs/2, Pgs/2, Pgs/2, Pgs/2, Pgs/2] ##Corregir
pii0_G = [Pgb,Pgb,Pgb,Pgb, Pgb,Pgb]
####################################################################################33
######################################################################################
prosumers = 6
PmasCP = range(prosumers)

#######################################################################################
#######################################################################################
#########################PROBLEMA DEL LADO DEL CONSUMIDOR##############################

def Welfarei(x,Pij,consumidores,generator,consumer,etha, lamdai,Gi,thetai):
    pii = x
    matriz = np.ones((consumidores, consumidores))
    np.fill_diagonal(matriz, 0)
    compe  = [sum(matriz[i][k]*pii[k]*sum([Pij[j][i] for j in generator]) for k in consumer) for i in consumer]
    #compe = [sum([Pij[j][i] for j in generator]) / sum(matriz[i][k]*np.log(np.abs(pii[k]))for k in consumer) for i in consumer]
    #Wi = [(sum([Pij[j][i] for j in generator])) /(np.log((ki[i]*np.abs(pii[i])+1))) - compe[i]/etha[i] for i in consumer] 
    Wi = [lamdai[i]*Gi[i]-thetai[i]*Gi[i]**2+(sum([Pij[j][i] for j in generator])) /(np.log((np.abs(pii[i])+1))) - compe[i]*etha[i] for i in consumer]
    return -sum(Wi)

def Costosj(x, Pij, a, b,generator,consumer,j):
    pii = x
    Costos = [a[j]*(sum([Pij[j][i] for i in consumer])**2)+b[j]*sum([Pij[j][i] for i in consumer])  for j in generator]
    re =[sum([pii[i]*Pij[j][i] for i in consumer]) for j in generator]
    R6 = Costos[j]-re[j]
    return -R6


def OptWi(Welfarei,pii0,bounds2,Pij,consumidores,generator,consumer,etha, lamdai,Di,thetai,Costosj):
    first_iteration = {}

    def callback(xk):
        if not first_iteration:  # Guarda la primera iteración
            first_iteration['x'] = np.copy(xk)
    inicio = time.time() #SLSQP trust-constr
    cons10 =   [{'type': 'ineq', 'fun': Costosj, 'args': (Pij, a, b,generator,consumer,j),} for j in generator]
    solution = minimize(Welfarei, pii0, method='SLSQP', bounds=bounds2, constraints=cons10 ,args=(Pij,consumidores,generator,consumer,etha, lamdai,Di,thetai), tol=1e-8)
    fin = time.time()
    duracion = fin - inicio
    piiSol = solution.x
    #piiSol = first_iteration['x']
    WelfarOpti = solution.fun
    return piiSol,-WelfarOpti,duracion #,first_iteration['x']
#########################################################################################
#########################################################################################
####################PROBLEMA DEL LADO DEL VENDEDOR#######################################

def Welfarejgen(x, Dj, a, b, thetaj, lamdaj, pii,generadores,consumidores,consumer,generator):
    Pij = x.reshape((generadores,consumidores))
    #Wj = [lamdaj[j]*Gj[j]-thetaj[j]*Gj[j]**2+sum([pii[i]*Pij[j][i] for i in consumer]) -a[j]*(sum([Pij[j][i] for i in consumer])**2)-b[j]*sum([Pij[j][i] for i in consumer]) for j in generator]
    Wj = [lamdaj[j]*Dj[j]-thetaj[j]*Dj[j]**2-sum([Pij[j][i]/(np.log(1+(pii[i]))) for i in consumer]) -a[j]*(sum([Pij[j][i] for i in consumer])**2)-b[j]*sum([Pij[j][i] for i in consumer]) for j in generator]
    return -sum(Wj)

#################RESTRICCIONES##########################################################
#########################################################################################
########################################################################################
def Restris1(x, j, Gj,generadores,consumidores,consumer):
    #Pij = x.reshape((consumidores, generadores))
    Pij = x.reshape((generadores,consumidores))
    R1 = sum([Pij[j][i] for i in consumer])-Gj[j]
    return -R1

def Restris2(x, i, Gj,Di,generadores,consumidores,generator):
    Pij = x.reshape((generadores,consumidores))
    if sum(Di) - sum(Gj)  > 0:
      R2 = sum([Pij[j][i] for j in generator]) -Di[i] #12
    else:
      R2 = -x
    return -R2

def Restris3(x, Di,i,generadores,consumidores,generator):
    Pij = x.reshape((generadores,consumidores))
    R4 = sum([Pij[j][i] for j in generator]) -Di[i] #### Ecuación 11
    return R4 ###

def Restris4(x,Gj,generadores,consumidores):
    Pij = x.reshape((generadores,consumidores))
    R5 = np.sum(Pij)-sum(Gj) ###Ecuación 13 para usar toda la generación
    return R5

# Definir el callback para capturar la primera iteración



def OptWj(Welfarejgen,sli0,bounds, a, b, thetaj, lamdaj, pii,generadores,consumidores,consumer,generator):
    first_iteration = {}

    def callback(xk):
        if not first_iteration:  # Guarda la primera iteración
            first_iteration['x'] = np.copy(xk)
        
    inicio = time.time() # 'trust-constr'
    cons1 = [{'type': 'ineq', 'fun': Restris1, 'args': (j, Gj,generadores,consumidores,consumer),} for j in generator]
    cons2 = [{'type': 'ineq', 'fun': Restris2, 'args': (i, Gj,Di,generadores,consumidores,generator)} for i in consumer]
    if sum(Di) - sum(Gj) <= 0:
        cons3 = [{'type': 'eq', 'fun': Restris3, 'args': (Di,i,generadores,consumidores,generator) } for i in consumer]
    else:
        cons3 = [{'type': 'eq', 'fun': Restris4,'args': (Gj,generadores,consumidores)}]
    solution = minimize(Welfarejgen, sli0, method='trust-constr', bounds=bounds, constraints=cons1+cons2+cons3,args=(Gj, a, b, thetaj, lamdaj, pii,generadores,consumidores,consumer,generator),tol=1e-8)
    fin = time.time()
    duracion = fin - inicio
    sli = solution.x
    #sli = [round(sli[j],2) for j in range(30)]
    profit = solution.fun
    Pijsolution = np.array(np.round(sli,4)).reshape((generadores,consumidores))
    #WelfareOpt = Welfarej_func(np.array(sli) , Gj, Yj, a, b, theta, lamda, npij,generadores,consumidores,consumer,generator)
    #print("Bienestar Vendedor",-profit, -WelfareOpt)
    print(Pijsolution)
    #print(first_iteration)
    #breakpoint()
    #Pijsolution = np.array(np.round(first_iteration['x'],4)).reshape((generadores,consumidores))
    return Pijsolution,-profit,duracion


###################################################################################################
###################################################################################################
###################################################################################################
########Verificación de restricciones condición inicial############################################
######Calculo de costos de generadores y recompenza recibida ######################################
def Costosk(Pij, a, b, pii):
    #npij = pii.reshape((generadores,consumidores))
    Costos = [a[j]*(sum([Pij[j][i] for i in consumer])**2)+b[j]*sum([Pij[j][i] for i in consumer]) for j in generator]
    re =[sum([pii[i]*Pij[j][i] for i in consumer]) for j in generator]
    cantidadcomprada = [sum([Pij[j][i] for j in generator]) for i in consumer]
    cantidadvendida = [sum([Pij[j][i] for i in consumer]) for j in generator]
    piprom = [pii[i]*sum([Pij[j][i] for j in generator]) for i in consumer]
    pjprom = [sum([pii[i]*Pij[j][i] for i in consumer]) for j in generator]
    return Costos, re, piprom, pjprom

def PruebaRestriG(Di, Pij,Gj, generator, consumer ):
    demandafinal= [sum([Pij[j][i] for j in generator]) for i in consumer]
    generacionfinal = [sum([Pij[j][i] for i in consumer]) for j in generator]
    generacionP2P = [Gj[j]  for j in generator]
    print(f'demanda final{demandafinal}, demandainicial{Di}')
    print(f"generación final{generacionfinal}", f"generacion iniial{generacionP2P}")
    return sum(demandafinal), sum(Di), sum(generacionfinal), sum(generacionP2P)

def CIprueba(x, bounds, a, b, thetaj, lamdaj, pii0,generadores,consumidores,consumer,generators):
    #Pij = x.reshape((generadores,consumidores))
    Wj = sum(x) ##Es como si guera el bienestar es como la suma de todas las variables 
    return Wj

PSOL=[]
CSOL=[]
BF = []
BI = []
BFC = []
BFG = []
BFC0 = []
BFG0 = []
CostosF = []
RecompenzaG= []

   
for t in range(24):
    #a= [0.089,24.7*0.067,0.067,0,0,0] ##Termica, Sol+W, Sol+W, eol, eol 10 veces los costos iniciales 
    #b= [448.6,696.1,795.5,800,838.1]
    #b= [52,24.7*32,32,47,37, 0,0]#[0.5, 0.8, 1, 1.3, 1.8]
    #a= [0.089,0.089+0.067,0.067,0,0,0] ##Termica, Sol+W, Sol+W, eol, eol 10 veces los costos iniciales 
        #b= [448.6,696.1,795.5,800,838.1]
    #b= [52,52+32,32,47,37,0,0]#[0.5, 0.8, 1, 1.3, 1.8]
    #t=18
    a= [0.089, 0.110, 0.069, 0, 0, 0] 

    b= [52, 58, 40, 37, 32, 0, 0]
    Glim = generation[t]
    Dopt = demand[t]
    Di = [Dopt[n] - Glim[n] for n in PmasCP if Glim[n] / Dopt[n] <= 1]
    Dj = [Dopt[n] for n in PmasCP if Glim[n]/Dopt[n]>=1 ]
    Gi = [Glim[n] for n in PmasCP if Glim[n]/Dopt[n]<=1]
    Gj = [Glim[n]-Dopt[n] for n in PmasCP if Glim[n]/Dopt[n]>=1 ]
    thetai = [theta[n] for n in PmasCP if Glim[n] / Dopt[n] <= 1]
    lamdai = [lamda[n] for n in PmasCP if Glim[n] / Dopt[n] <= 1]
    thetaj = [theta[n] for n in PmasCP if Glim[n]/Dopt[n]>=1 ] 
    lamdaj = [lamda[n] for n in PmasCP if Glim[n]/Dopt[n]>=1 ]
    etha = [etha0 [n] for n in PmasCP if Glim[n] / Dopt[n] <= 1]
    consumidores =len(Di)
    consumer = range(consumidores)
    generadores = len(Gj)
    generator = range(generadores)
    bounds = [(0, (1000*Gj[j])) for n in consumer for j in generator ]
    bounds2 = [(Pgb,Pgs) for _ in consumer]   
    
    pii0 = [precioin[n] for n in PmasCP if Glim[n] / Dopt[n] <= 1]
    a =  [a[n] for n in PmasCP if Glim[n]/Dopt[n]>=1 ] 
    b =  [b[n] for n in PmasCP if Glim[n]/Dopt[n]>=1 ]
    
    if Gj==[] :
        sli0 = 0*np.ones((generadores*consumidores))
        BienestarjG = Welfarejgen(sli0,Gj, a, b, thetaj, lamdaj, pii0,generadores,consumidores,consumer,generator)
        BienestariG = Welfarei(pii0,sli0,consumidores,generator,consumer,etha, lamdai,Di,thetai)
       #Cantidades_matris2 = np.array([[0 if Glim[i] == 0 or Glim[n]/Dopt[n]>=1 else Glim[n] for i in PmasCP] for n in PmasCP],dtype=float) 
        Cantidades_matris2 = np.zeros((prosumers,prosumers))
        for i, (g, d) in enumerate(zip(Glim, Dopt)):
            if g/d > 1:
                column_matrix = np.ones((prosumers,))
                column_matrix[i] = d 
                Cantidades_matris2[:,i] = column_matrix
            else:
                column_matrix = np.zeros((prosumers,))
                column_matrix[i] = g
                Cantidades_matris2[:,i] = column_matrix
        for i, (g, d) in enumerate(zip(Glim, Dopt)):   
            if  Cantidades_matris[i, i] == d:
                    row_matrix = np.zeros(6,)
                    row_matrix[i] = d
                    Cantidades_matris2[i, :] = row_matrix
       
        Precios_soluciones = [Pgs for n in PmasCP]

        PSOL.append(Precios_soluciones)
        CSOL.append(Cantidades_matris2.flatten())
        BFG.append(-BienestarjG)
        BFG0.append(-BienestarjG)
        BFC0.append(-BienestariG)
        BFC.append(-BienestariG)
        BF.append(-BienestarjG -BienestariG)
        BI.append(-BienestarjG-BienestariG)
    elif Di == []:
        Precios_soluciones = [0 for n in PmasCP]
        sli0 = 0*np.ones((generadores*consumidores))
        BienestarjG = Welfarejgen(sli0,Gj, a, b, thetaj, lamdaj, pii0,generadores,consumidores,consumer,generator)
        BienestariG = Welfarei(pii0,sli0,consumidores,generator,consumer,etha, lamdai,Di,thetai)
       #Cantidades_matris2 = np.array([[0 if Glim[i] == 0 or Glim[n]/Dopt[n]>=1 else Glim[n] for i in PmasCP] for n in PmasCP],dtype=float) 
        Cantidades_matris2 = np.zeros((prosumers,prosumers))
        for i, (g, d) in enumerate(zip(Glim, Dopt)):
            if g/d > 1:
                column_matrix = np.ones((prosumers,))
                column_matrix[i] = d 
                Cantidades_matris2[:,i] = column_matrix
            else:
                column_matrix = np.zeros((prosumers,))
                column_matrix[i] = g
                Cantidades_matris2[:,i] = column_matrix
        for i, (g, d) in enumerate(zip(Glim, Dopt)):   
            if  Cantidades_matris[i, i] == d:
                    row_matrix = np.zeros(6,)
                    row_matrix[i] = d
                    Cantidades_matris2[i, :] = row_matrix
        PSOL.append(Precios_soluciones)
        CSOL.append(Cantidades_matris2.flatten())
        BFG.append(-BienestarjG)
        BFG0.append(-BienestarjG)
        BFC0.append(-BienestariG)
        BFC.append(-BienestariG)
        BF.append(-BienestarjG -BienestariG)
        BI.append(-BienestarjG-BienestariG)
        
    else:
        ####################################################################################
        ####################################################################################
        sli0 = 0*np.ones((generadores*consumidores))
        Pij00 = OptWj(CIprueba,sli0,bounds, a, b, thetaj, lamdaj, pii0,generadores,consumidores,consumer,generator)
        print('PRUEBA SOLUCIONES POTENCIA', Pij00)
        DF, DI, GF, GI = PruebaRestriG(Di, Pij00[0],Gj, generator, consumer)
        #print("condiciones iniciales",DF,DI,GF,GI)
        Pij0 = Pij00[0].flatten()
        #print(Pij0, "Potencias iniciales decididas por los generadores")
        print('PRECIOS INICIALES', pii0)
        #####################################################################################
        #####################################################################################
        BienestarC = []
        Bienestarj = []
        Bienestari = []
        SolutionPrice = []
        SolutionPij = []
        BienestarC = []
        Bienestarj0 = Welfarejgen(Pij0,Dj, a, b, thetaj, lamdaj, pii0,generadores,consumidores,consumer,generator)
        print("Bienestar inicial generadores",-Bienestarj0)
        Bienestari0 = Welfarei(pii0,Pij00[0],consumidores,generator,consumer,etha, lamdai,Gi,thetai)
        print("Bienestar inicial consumidor",-Bienestari0)
        Bienestar00 = -Bienestarj0 -Bienestari0
        print("bienestar de la comunidad", Bienestar00)
        BienestarC.append(Bienestar00)
        Pijsolution = Pij00[0]
        for i in range(10): 
            piiSol, WelfarOpti,duracion= OptWi(Welfarei,pii0,bounds2,Pijsolution,consumidores,generator,consumer,etha, lamdai,Di,thetai,Costosj)
            print("precios", piiSol)
            Bienestari0 = -Welfarei(pii0,Pijsolution,consumidores,generator,consumer,etha, lamdai,Gi,thetai)
            Wiopt = -Welfarei(piiSol,Pijsolution,consumidores,generator,consumer,etha, lamdai,Gi,thetai)
            Pijsolution,profit,duracion = OptWj(Welfarejgen,Pij0,bounds, a, b, thetaj, lamdaj, piiSol,generadores,consumidores,consumer,generator)
            Bienestarj0 = -Welfarejgen(Pij00[0],  Dj, a, b, thetaj, lamdaj, piiSol,generadores,consumidores,consumer,generator)
            Wjopt = -Welfarejgen(Pijsolution,  Dj, a, b, thetaj, lamdaj, piiSol,generadores,consumidores,consumer,generator)
            
            print("Bienestar de la comunidad", (Wiopt+Wjopt))
            BienestarC.append((Wiopt+Wjopt))
        
            if Wiopt+Wjopt >= Bienestarj0 +  Bienestari0 and i>=2 :
                break
        #breakpoint()
        print('Pijsolution',Pijsolution)
        Precios_soluciones = []
        Precios_iniciales = [precioin[n] if Glim[n] / Dopt[n] <= 1 else 0 for n in PmasCP ]
        index_piiSol = 0
        Pijsolution = np.transpose(Pijsolution)
        
        #Precios_soluciones = [0 if Precios_soluciones[i] == 0 else piiSol[index_piiSol] for i in PmasCP if (index_piiSol := index_piiSol+1) or True]
        for i in PmasCP:
            if Precios_iniciales[i] == 0:
                Precios_soluciones.append(0)
            else:
                Precios_soluciones.append(piiSol[index_piiSol])
                index_piiSol += 1
        print(Precios_soluciones)
        

        Cantidades_matris = np.zeros((prosumers,prosumers))
        for i, (g, d) in enumerate(zip(Glim, Dopt)):
            if g/d > 1:
                column_matrix = np.ones((prosumers,))
                column_matrix[i] = d 
                Cantidades_matris[:,i] = column_matrix
            else:
                column_matrix = np.zeros((prosumers,))
                column_matrix[i] = g
                Cantidades_matris[:,i] = column_matrix
        for i, (g, d) in enumerate(zip(Glim, Dopt)):   
            if  Cantidades_matris[i, i] == d:
                    row_matrix = np.zeros(6,)
                    row_matrix[i] = d
                    Cantidades_matris[i, :] = row_matrix
               #breakpoint()
        #Cantidades_matris = np.array([[Glim[n] if Glim[n]-Dopt[n]<=0 else 1 for n in PmasCP] for i in PmasCP],dtype=float)
        #Cantidades_matris = np.array([[Glim[n]  for n in PmasCP if Glim[n]-Dopt[n]<=0] for i in PmasCP],dtype=float)
        print('Cantidades_matris',Cantidades_matris)
        replacement_index = 0
        replacement_values = Pijsolution.flatten()
        Cantidades_matris2 = np.copy(Cantidades_matris)
        Cantidades_iniciales = [0 if Glim[n]/Dopt[i]<=1 else Dopt[n] for n in PmasCP]
        # Reemplazar los `1`s en `matrix` con los valores de `replacement_matrix`
        for i in PmasCP:
            #Cantidades_matris2[i, i] = Cantidades_iniciales[i]
            for j in PmasCP:
                if Cantidades_matris[i, j] == 1:
                    # Si hay más valores en replacement_matrix para usar
                    if replacement_index < len(replacement_values):
                        Cantidades_matris2[i, j] = replacement_values[replacement_index]
                        replacement_index += 1

        print(Cantidades_matris2)
        #Costos, re, piprom, pjprom = Costosk()
    #breakpoint()
        #Costos, re, piprom, pjprom = Costosk(Pijsolution.T, a, b, piiSol)
        #print("costos",Costos)
        #print("recom",re)
        PSOL.append(Precios_soluciones)
        CSOL.append(Cantidades_matris2.flatten())
        BFG.append(profit)
        BFG0.append(-Bienestarj0)
        BFC0.append(-Bienestari0)
        BFC.append(WelfarOpti)
        BF.append((Wiopt+Wjopt))


        BI.append(Bienestar00)
    #CostosF.append(Costos)
    #RecompenzaG.append(re)
P_porP = np.transpose(PSOL)
C_porP =np.transpose(CSOL)
print(P_porP)
print(C_porP)
np.concatenate((C_porP, P_porP)) 

labels = ['P1G1', 'P1G2','P1G3', 'P1G4','P1G5', 'P1G6','P2G1', 'P2G2','P2G3','P2G4','P2G5','P2G6','P3G1', 'P3G2','P3G3','P3G4','P3G5','P3G6','P4G1', 'P4G2','P4G3','P4G4','P4G5','P4G6','P5G1', 'P5G2','P5G3','P5G4','P5G5','P5G6','P6G1', 'P6G2','P6G3','P6G4','P6G5','P6G6','$P1','$P2','$P3','$P4','$P5','$P6']
df2 = pd.DataFrame(np.concatenate((C_porP, P_porP)) , index=labels)
df3 = df2.transpose()
#labels2 = ['CostosG1','CostoG2','CostoG3','GananciaG1','GananciaG2','GananciaG3' ]
#df4 = pd.DataFrame(np.concatenate((CostosF, RecompenzaG)) , index=labels2)
dffinal = pd.concat((df,df3), axis=1)
#dffinal = df3
dffinal['Pext'] = (dffinal[['Glim1', 'Glim2', 'Glim3','Glim4','Glim5','Glim6']].sum(axis=1) - dffinal[['D*1', 'D*2', 'D*3','D*4','D*5','D*6']].sum(axis=1)).apply(lambda x: x if x > 0 else 0)
dffinal['Pint'] = (dffinal[['D*1', 'D*2', 'D*3','D*4','D*5','D*6']].sum(axis=1) - dffinal[['Glim1', 'Glim2', 'Glim3','Glim4','Glim5','Glim6']].sum(axis=1)).apply(lambda x: x if x > 0 else 0)
#dffinal['GananciaP1'] = dffinal['P2G1']*dffinal['$P2'] + dffinal['P3G1']*dffinal['$P3'] + dffinal['P4G1']*dffinal['$P4'] 
#a= [0.089,24.7*0.067,0.067,0,0,0] ##Termica, Sol+W, Sol+W, eol, eol 10 veces los costos iniciales 
#b= [448.6,696.1,795.5,800,838.1]
#b= [52,24.7*32,32,47,37, 0,0]#[0.5, 0.8, 1, 1.3, 1.8]
a= [0.089, 0.110, 0.069, 0, 0, 0] 

b= [52, 58, 40, 37, 32, 0, 0]
"""
dffinal['CostosG1'] = a[0]*(dffinal[['P2G1','P3G1','P4G1']].sum(axis=1))**2 +b[0]*(dffinal[['P2G1','P3G1','P4G1']].sum(axis=1))
dffinal['GananciaG1'] =  dffinal['P2G1']*dffinal['$P2']  + dffinal['P3G1']*dffinal['$P3'] + dffinal['P4G1']*dffinal['$P4'] 
dffinal['GaG1'] = dffinal['GananciaG1']-dffinal['CostosG1']
dffinal['CostosG2'] = a[1]*(dffinal[['P1G2','P3G2','P4G2']].sum(axis=1))**2 +b[1]*(dffinal[['P1G2','P3G2','P4G2']].sum(axis=1))
dffinal['GananciaG2'] =  dffinal['P1G2']*dffinal['$P1'] + dffinal['P3G2']*dffinal['$P3']  +  dffinal['P4G2']*dffinal['$P4'] 
dffinal['GaG2'] = dffinal['GananciaG2']-dffinal['CostosG2']
dffinal['CostosG3'] = a[2]*(dffinal[['P1G3','P2G3','P4G3']].sum(axis=1))**2 +b[2]*(dffinal[['P1G3','P2G3','P4G3']].sum(axis=1))
dffinal['GananciaG3'] =  dffinal['P1G3']*dffinal['$P1'] + dffinal['P2G3']*dffinal['$P2']  +  dffinal['P4G3']*dffinal['$P4'] 
dffinal['GaG3'] = dffinal['GananciaG3']-dffinal['CostosG2']
dffinal['CostosG4'] = a[3]*(dffinal[['P1G4','P2G4','P3G4']].sum(axis=1))**2 +b[3]*(dffinal[['P1G4','P2G4','P3G4']].sum(axis=1))
dffinal['GananciaG4'] =  dffinal['P1G4']*dffinal['$P1'] + dffinal['P2G4']*dffinal['$P2']  +  dffinal['P3G4']*dffinal['$P3'] 
dffinal['GaG4'] = dffinal['GananciaG4']-dffinal['CostosG4']
dffinal['BF'] = BF
dffinal['BI'] = BI
dffinal['Ganancia comunidad'] = dffinal['BF']-dffinal['BI']
dffinal['BFG'] = BFG
dffinal['BFG0'] = BFG0
dffinal['BFC'] = BFC
dffinal['BFC0'] = BFC0

dffinal['RP1'] = (dffinal['P1G4']+dffinal['P1G2']+dffinal['P1G3'])*Pgs - (dffinal['P1G4']+dffinal['P1G2'] +dffinal['P1G3'] )*dffinal['$P1'] 
dffinal['RP2'] = (dffinal['P2G4']+dffinal['P2G1']+dffinal['P2G3'])*Pgs - (dffinal['P2G4'] +dffinal['P2G1'] +dffinal['P2G3'])*dffinal['$P2'] 
dffinal['RP3'] = (dffinal['P3G4']+dffinal['P3G2']+dffinal['P3G1'])*Pgs - (dffinal['P3G4'] +dffinal['P3G2'] +dffinal['P3G1'] )*dffinal['$P3'] 
dffinal['RP4'] = (dffinal['P4G1']+dffinal['P4G2']+dffinal['P4G3'])*Pgs - (dffinal['P4G1']+dffinal['P4G2']+dffinal['P4G3'])*dffinal['$P4'] 

dffinal['GTP1'] = dffinal['RP1']+dffinal['GaG1']
dffinal['GTP2'] = dffinal['RP2']+dffinal['GaG2']
dffinal['GTP3'] = dffinal['RP3']+dffinal['GaG3']
dffinal['GTP4'] = dffinal['RP4']+dffinal['GaG4']
"""
dffinal['BF'] = BF
dffinal['BI'] = BI
dffinal['Ganancia comunidad'] = dffinal['BF']-dffinal['BI']
dffinal['BFG'] = BFG
dffinal['BFG0'] = BFG0
dffinal['BFC'] = BFC
dffinal['BFC0'] = BFC0
pd.set_option('display.max_columns', None) 
#print(dffinal)
dffinal.to_excel('Prueba3Prosumidores6.xlsx', index=False)
#for i in PmasCP:

 #   matrix[i, i] = vector[i]

#####
dP2PP1 = dffinal[['P1G1','P1G2', 'P1G3', 'P1G4', 'P1G5', 'P1G6']].copy()
matrizP1 = dP2PP1.to_numpy()
dP2PP2 = dffinal[['P2G1','P2G2','P2G3','P2G4','P2G5','P2G6']].copy()
matrizP2 = dP2PP2.to_numpy()
dP2PP3 = dffinal[['P3G1','P3G2','P3G3','P3G4','P3G5','P3G6']].copy()
matrizP3 = dP2PP3.to_numpy()
dP2PP4 = dffinal[['P4G1','P4G2','P4G3','P4G4','P4G5','P4G6']].copy()
matrizP4 = dP2PP4.to_numpy()
dP2PP5 = dffinal[['P5G1','P5G2','P5G3','P5G4','P5G5','P5G6']].copy()
matrizP5 = dP2PP5.to_numpy()
dP2PP6 = dffinal[['P6G1','P6G2','P6G3','P6G4','P6G5','P6G6']].copy()
matrizP6 = dP2PP6.to_numpy()

Precios = dffinal[['$P1','$P2','$P3','$P4','$P5','$P6']].copy()
Precios = Precios.to_numpy()
import matplotlib.pyplot as plt
#import seaborn as sns
generadores = [f'G{i+1}' for i in range(6)]
consumidores = [f'C{i+1}' for i in range(6)]
for t in range(24):
    Pijsolution =np.transpose(np.array([matrizP1[t],matrizP2[t],matrizP3[t],matrizP4[t],matrizP5[t],matrizP6[t]]))
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[8, 1])  # 2 filas, 1 columna, diferentes alturas

    # Primer subplot (más alto)
    ax1 = fig.add_subplot(gs[0, 0])
# Primera gráfica
    ax1.imshow(Pijsolution, cmap='viridis', interpolation='nearest')
    for i in range(Pijsolution.shape[0]):
        for j in range(Pijsolution.shape[1]):
            ax1.text(j, i, f'{Pijsolution[i, j]:.2f}', ha='center', va='center', color='white')
    ax1.set_xticks(np.arange(Pijsolution.shape[1]))
    ax1.set_xticklabels(np.arange(1, Pijsolution.shape[1] + 1))
    ax1.set_yticks(np.arange(Pijsolution.shape[0]))
    ax1.set_yticklabels(np.arange(1, Pijsolution.shape[0] + 1))
    ax1.set_xlabel('Prosumidores (Compradores)')
    ax1.set_ylabel('Prosumidores (Generadores)')
    ax1.set_title(f'Quantities of power of H = {t+1} ')
    fig.colorbar(ax1.imshow(Pijsolution, cmap='viridis', interpolation='nearest'),shrink=0.8, ax=ax1, orientation='vertical')

    
    piiSol_matrix = np.array([Precios[t]])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(piiSol_matrix, cmap='viridis', aspect='auto')
    for i in range(piiSol_matrix.shape[1]):
        ax2.text(i, 0, f'{Precios[t][i]:.2f}', ha='center', va='bottom', color='white')
    ax2.set_xticks(np.arange(piiSol_matrix.shape[1]))
    ax2.set_xticklabels(np.arange(1, piiSol_matrix.shape[1] + 1))
    ax2.set_yticks([])
    ax2.set_xlabel('Buyer')
    ax2.set_ylabel('Prices')
    #ax2.set_title('Price decided by the buyer')
    fig.colorbar(ax2.imshow(piiSol_matrix, cmap='viridis', aspect='auto'), ax=ax2, orientation='vertical')
    plt.show()
    #breakpoint()

