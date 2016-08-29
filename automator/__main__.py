
def main():
    from matplotlib import rc
    rc('backend', qt4="PySide")
    
    import automator, sys
    try:
        mainGui = automator.Automator()
    except SystemExit:
        del mainGui
        sys.exit()
        exit
        
# Instantiate a class
if __name__ == '__main__': # Windows multiprocessing safety
    main()
