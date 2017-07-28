#Importar tablas FOREX
{if(!require(sqldf)){
  install.packages("sqldf")
  library(sqldf)}
}

setwd("U:/Users/dcaramu/Desktop/FOREX")

for (i in 1:7){assign(paste0("nom_tabla",as.character(i)),sprintf("DAT_ASCII_AUDUSD_T_20170%i.csv",i))}
for (i in 1:7){assign(paste0("AUDUSD",as.character(i)),read.table(get(paste0("nom_tabla",as.character(i))),header=FALSE, sep = ","))}
for (i in 1:7){rm(list = paste0("nom_tabla",as.character(i)))}
AUDUSD=rbind(AUDUSD1,AUDUSD2,AUDUSD3,AUDUSD4,AUDUSD5,AUDUSD6,AUDUSD7)
for (i in 1:7){rm(list = paste0("AUDUSD",as.character(i)))}

AUDUSD1=data.frame(datetimeold=AUDUSD$V1, datetime=substr(AUDUSD$V1,1, 13), bid=AUDUSD$V2, ask=AUDUSD$V3)
rm(AUDUSD)
AUDUSD=sqldf('select datetime, max(bid) as bid, max(ask) as ask from AUDUSD1 group by datetime') 
rm(AUDUSD1)

EURUSD1=data.frame(datetimeold=EURUSD$V1, datetime=substr(EURUSD$V1,1, 13), bid=EURUSD$V2, ask=EURUSD$V3)
rm(EURUSD)
EURUSD=sqldf('select datetime, max(bid) as bid, max(ask) as ask from EURUSD1 group by datetime') 
rm(EURUSD1)

GBPUSD1=data.frame(datetimeold=GBPUSD$V1, datetime=substr(GBPUSD$V1,1, 13), bid=GBPUSD$V2, ask=GBPUSD$V3)
rm(GBPUSD)
GBPUSD=sqldf('select datetime, max(bid) as bid, max(ask) as ask from GBPUSD1 group by datetime') 
rm(GBPUSD1)

USDJPY1=data.frame(datetimeold=USDJPY$V1, datetime=substr(USDJPY$V1,1, 13), bid=USDJPY$V2, ask=USDJPY$V3)
rm(USDJPY)
USDJPY=sqldf('select datetime, max(bid) as bid, max(ask) as ask from USDJPY1 group by datetime') 
rm(USDJPY1)

FOREX=sqldf('select a.datetime, a.bid as AUDUSD_bid, a.ask as AUDUSD_ask,
            b.bid as EURUSD_bid, b.ask as EURUSD_ask,
            c.bid as GBPUSD_bid, c.ask as GBPUSD_ask,
            d.bid as USDJPY_bid, d.ask as USDJPY_ask 
            from AUDUSD a 
            inner join EURUSD b on a.datetime=b.datetime
            inner join GBPUSD c on a.datetime=C.datetime
            inner join USDJPY d on a.datetime=d.datetime
           ')
##FOREX=data.frame(DATE=FOREX$date, USDEUR= format(round(1/FOREX$EURUSD, 4), nsmall = 4), USDAUD= format(round(1/FOREX$AUDUSD, 4), nsmall = 4), USDGBP= format(round(1/FOREX$GBPUSD, 4), nsmall = 4), USDJPY=FOREX$USDJPY)

FOREX = data.frame(order=seq.int(nrow(FOREX)), FOREX)
write.table(FOREX, "U:/Users/dcaramu/Desktop/FOREX/forex.txt", sep=";", row.names = FALSE)
