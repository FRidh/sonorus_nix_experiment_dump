#import argparse
#from h5store import jsonsave
#from sonorus import DB_BLAZE
#from blaze import Data
#from odo import odo

#INCLUDE = 1

#def get_events(nevents, aircraft, operation):
    #"""
    #:param aircraft: List of aircraft names
    #:param operation: List of operation names

    #"""
    #INCLUDE = 1

    #d = Data(DB_BLAZE)
    #e = d.event

    ## Selection of events
    #s = e[e.include==INCLUDE and
          #e.operation.isin(operation) and
          #e.aircraft.isin(aircraft)]

    #s = odo(s, pd.Series)

    #s.sample(n=nevents)


    #pass



#def main():

    #argparse.ArgumentParser()
    #parser.add_argument("settings", type=str)
    #parser.add_argument("file_out", type=str)
    #args = parser.parse_args()

    ## List of events
    #events = get_events()

    #jsonsave(args.file_out, events)




#if __name__ == '__main__':
    #main()
