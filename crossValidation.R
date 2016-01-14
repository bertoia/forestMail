library(xgboost)
library(caret)
#save(target.train,file = "target.train.rda" )
#load("target.train.rda")

##create cross validation indexes##
index = createFolds(target.train,k=10,list = F)

##Create xgb DMatrix
for (ii in 1:10) {
  gc()
  xgb.data.1       = xgb.DMatrix(data.matrix(train.p[index!=ii,]), label=target.train[index!=ii], missing = NA)
  gc()
  xgb.validation.1 = xgb.DMatrix(data.matrix(train.p[index==ii,]), label=target.train[index==ii], missing = NA)
  gc()
  ##Save DMatrix
  xgb.DMatrix.save(xgb.data.1,fname = paste("../forestMailAWS/xgb.data.",ii,".data",sep = ""))
  xgb.DMatrix.save(xgb.validation.1,fname = paste("../forestMailAWS/xgb.validation.",ii,".data",sep = ""))
}


/*************************************
* Lab 5 Exercise 3
* Name: VINCENT SENG BOON CHIN
* Matric No: A0121501E
* Lab Group: 5
*************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h> //for memcpy()

/**************** Endian Conversion *************/

//Commment off the following define for Big endian machine
// e.g. Solaris (SPARC)
//Uncomment for Little Endian machine e.g. intel
#define LITTLEENDIAN

int convertEndian(int w)
{
#ifdef LITTLEENDIAN
    int result;

    result = (w >> 24) & 0x000000ff;
    result |= (w >> 8) & 0x0000ff00;
    result |= (w << 8) & 0x00ff0000;
    result |= (w << 24) & 0xff000000;
    return result;
#else
    return w;
#endif
}

/**************** General Definition *************/
#define LOGICAL_MEM_SIZE 256    //256 words logical memory space
#define PAGESIZE 8              // a tiny example
#define LOGICALPAGES (LOGICAL_MEM_SIZE/PAGESIZE)

typedef struct {
    int loc[PAGESIZE];  //locations in a page
} page;

void printPage( page* );
//Print the content of page

/**************** Secondary Storage **************/

typedef struct {
    int swapFileFD;     //file descriptor for swap file
    int nSwapPage;      //maximum swap pages in swap file
} swapFile;

int initSwapFile(swapFile* sf, char* fileName);
//Initialize the swap file "sf" by using the file "fileName"
//Assumption:
//   - the number of swap pages = logical memory size
//Return:
//  0 = failure
//  1 = success, with sf initialized properly

int readSwapPage(swapFile*, page*, int);
int writeSwapPage(swapFile*, page*, int);

void printSwapFile(swapFile* sf);

/**************** Physical Memory ****************/

#define PHYSICALFRAMES 8
typedef struct {
    int nFrame;     //maximum number of physical frames
    page frameArr[PHYSICALFRAMES];  //model the memory as an array of frames
} memory;

void initMemory( memory* );

int readOneFrame( memory*, page*, int );
int writeOneFrame( memory*, page*, int );

int readOneWord( memory*, int, int, int* );
int writeOneWord( memory*, int, int, int );

void printMemory( memory* );

/**************** Operating System ***************/

//Constants for the whereBit
//  show where a logical page is
#define IN_MEMORY 0
#define IN_DISK 1

//Constatns for teh validBit
//  show whether a logical page# is vin range
#define INVALID 0
#define VALID 1

typedef struct {   //Page Table Entry 
    int frameNumber;    
    char whereBit;
    char validBit;
} pte;

void printPTE( pte* );

typedef struct {
    pte pageTable[ LOGICALPAGES ];
    int frameUsageTable[ PHYSICALFRAMES ];
    memory *physicalMem;
    swapFile *secStorage;
} OS;               //Memory information kept in OS


void initOperatingSystem( OS*, memory*, swapFile* );

int loadProgram( OS*, char*);

int locatePTE( OS*, int, int*, pte* );

int handlePageFault( OS*, int, int*);

void printOperatingSystem( OS* );


/**************** CPU ****************************/

#define DISABLED 0
#define ENABLED 1

#define TLBSIZE 4
typedef struct{   //TLB Entry
    pte pageTableEntry;    //Contains one page table entry
    int pteNumber;         //The page number of the pte
    char enableBit;        //Whether this TLB entry is valid or not
} tlbe;

typedef struct {
    tlbe TLB[4];
    memory *physicalMem;
    OS *operatingSystem;
} cpu;

void initCPU( cpu*, memory*, OS* );

int searchTLB( tlbe*, int );

int findFrameNumber( cpu*, int );

int readMemAccess( cpu*, int, int, int*);

int writeMemAccess( cpu*, int, int, int);

void printCPU( cpu* );


/**************** Other ****************************/

int main()
{
    
    char fileName[21];
    int i, numAccess, accessType, pageNum, offset, value;

    memory RAM;  
    OS myOS;
    cpu theCPU;
    swapFile secondaryStorage;

    //Check for 32-bit vs 64-bit environment.
    printf("Integer is %d bytes. Should be 4-bytes (32-bit).\n", sizeof(int));
    if (sizeof(int) != 4 ){
        printf("Non 32-bit environment deteced, aborting.\n");
        printf("Please recompile with \"-m32\" flag.\n");
        return 1;
    }

    //Initialize swap file
    printf("Enter swap file name: ");
    scanf("%s", fileName);

    if (!initSwapFile( &secondaryStorage, fileName ) ){
        printf("Cannot initialize swap file!\n");
        return 1;
    } 

    //Initialize the main memory
    initMemory(&RAM);


    //Initialize operating system and load program
    printf("Enter program file to run: ");
    scanf("%s", fileName);

    initOperatingSystem( &myOS, &RAM, &secondaryStorage );
    if (!loadProgram( &myOS, fileName)){
        printf("Cannot load program!\n");

    }

    //Initialize the simple CPU 
    initCPU( &theCPU, &RAM, &myOS );

    
    //Memory access
    printf("Enter number of memory access: ");
    scanf("%i", &numAccess );


    for( i = 0; i < numAccess; ){
        scanf("%i", &accessType);
        switch( accessType) {
            case 1:
                scanf("%i %i", &pageNum, &offset);
                printf("Read %i.%i: ", pageNum, offset );
                if( readMemAccess( &theCPU, pageNum, offset, &value)){
                    printf("%i\n", value);
                } else {
                    printf("failed\n");
                }
                i++;
                break;
            case 2:
                scanf("%i %i %i", &pageNum, &offset, &value);
                printf("Write %i -> %i.%i : ", value, pageNum, offset );
                if( writeMemAccess( &theCPU, pageNum, offset, value)){
                    printf("ok\n");
                } else {
                    printf("failed\n");
                }
                i++;
                break;
        //Options 3 to 6 are for debug purpose only
            case 3:
                printCPU( &theCPU );
                break;
            case 4:
                printOperatingSystem( &myOS );
                break;
            case 5:
                printMemory( &RAM );
                break;
            case 6:
                printSwapFile( &secondaryStorage );
                break;
        }
    }
    // TODO:
    //    Print out the following statistics here:
    //    1. Total Number of access
    //    2. Percentage of TLB-Miss (2 place of precision)
    //    3. Percentage of Page-Fault (2 place of precision)       

    return 0;
}

/**************** General Definition *************/
void printPage( page* pagePtr)
{
    int i;

    for (i = 0; i < PAGESIZE; i++){
        printf("%i ", pagePtr->loc[i]);
        if (!((i+1)%4))     //print 4 per line
            printf("\n");
    }
}

void fillWithZero( page* pagePtr)
{
    int i;

    for( i = 0; i < PAGESIZE; i++)
        pagePtr->loc[i] = 0;
}

/**************** Secondary Storage **************/
int initSwapFile(swapFile* sf, char* fileName)
{
    int fdOut, i;
    page buffer;    //declare a buffer of a page size

    //Open the file "fileName" to prepare it as swap space
    //Create a new file if necessary
    // File opened for read + write.
    // The 3rd parameter is useful only to set permission of a new file
    fdOut = open( fileName, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );
    if (fdOut < 0){
        return 0;
    }

    fillWithZero( &buffer );

    for( i = 0; i < LOGICALPAGES; i++)
        write( fdOut, &buffer, sizeof(page));

    sf->swapFileFD = fdOut;
    sf->nSwapPage = LOGICALPAGES;

    return 1;

}

int readSwapPage(swapFile* sf, page* onePage, int swapPageNum)
{
    if (swapPageNum >= sf->nSwapPage)
        return 0;

    lseek( sf->swapFileFD, sizeof(page)*swapPageNum, SEEK_SET);    
    return read(sf->swapFileFD, onePage, sizeof(page));
}

int writeSwapPage(swapFile* sf, page* onePage, int swapPageNum)
{
    if (swapPageNum >= sf->nSwapPage)
        return 0;

    lseek( sf->swapFileFD, sizeof(page)*swapPageNum, SEEK_SET);    
    return write(sf->swapFileFD, onePage, sizeof(page));

}

void printSwapFile( swapFile* sf )
{
    page tempPage;
    int i;

    printf("Swap File with %i Swap Pages\n", LOGICALPAGES);
    
    for(i = 0; i < LOGICALPAGES; i++){  
        readSwapPage( sf, &tempPage, i);
        printf("----- Swap Page %i -----\n",i);
        printPage( &tempPage);
    }
    printf("\n");

}

/**************** Physical Memory ****************/
void initMemory( memory* mem )
{
    int i;

    mem->nFrame = PHYSICALFRAMES;   
    
    for (i = 0; i < mem->nFrame; i++){
        fillWithZero(&(mem->frameArr[i]));
    }

}

int readOneFrame( memory* mem, page* onePage, int pageNum )
{
    if (pageNum >= mem->nFrame)
        return 0;
    
    memcpy(onePage, &(mem->frameArr[pageNum]), sizeof(page));

    return 1;
}

int writeOneFrame( memory* mem, page* onePage, int pageNum )
{
   if (pageNum >= mem->nFrame)
        return 0;
    
    memcpy(&(mem->frameArr[pageNum]), onePage, sizeof(page));

    return 1;

}

int readOneWord( memory* mem, int pageNum, int offset, int* value)
{
    if (pageNum >= mem->nFrame || offset >= PAGESIZE )
        return 0;

    *value = mem->frameArr[pageNum].loc[offset];
    return 1;

}

int writeOneWord( memory* mem, int pageNum, int offset, int value )
{
    if (pageNum >= mem->nFrame || offset >= PAGESIZE )
        return 0;

    mem->frameArr[pageNum].loc[offset] = value;
    return 1;
}

void printMemory( memory* mem )
{
    int i;

    printf("Physical Memory with %i frames\n", mem->nFrame);
    for (i = 0; i < mem->nFrame; i++){
        printf("----- Frame %i -----\n",i);
        printPage(&(mem->frameArr[i]));
    }
    printf("\n");

}

/**************** Operating System ***************/
void printPTE( pte* pageTableEntry )
{
    if (pageTableEntry->validBit == VALID){

        if (pageTableEntry->whereBit == IN_MEMORY){
            printf(" RAM");
        } else {
            printf("DISK");
        }
        printf("[%i]\n", pageTableEntry->frameNumber);
    } else {
        printf("INVALID\n");
    }     
}

void initOperatingSystem( OS* os, memory* mem, swapFile* sf)
{
    int i;

    os->physicalMem = mem;
    os->secStorage = sf;

    for ( i = 0; i < LOGICALPAGES; i++){
        os->pageTable[i].validBit = INVALID;
    }     

    for ( i = 0; i < PHYSICALFRAMES; i++){
        os->frameUsageTable[i] = -1;
    }

}

int loadProgram( OS* os, char* programFileName)
{
    int fdIn, programSize, programData, i;
    int pageNum;
    page tempPage;


    fdIn = open( programFileName, O_RDONLY );
    if ( fdIn < 0 )
        return 0;

    fillWithZero( &tempPage);

    read( fdIn, &programSize, sizeof(int) );
    programSize = convertEndian( programSize );

    for (i = 0, pageNum = 0; i < programSize; i++){
        read( fdIn, &programData, sizeof(int) );
        programData = convertEndian( programData );
        tempPage.loc[i%PAGESIZE] = programData;
        if ((i+1)% PAGESIZE == 0) {        //one page is full
            writeSwapPage( os->secStorage, &tempPage, pageNum);

            os->pageTable[pageNum].frameNumber = pageNum;
            os->pageTable[pageNum].validBit = VALID;
            os->pageTable[pageNum].whereBit = IN_DISK;  
            pageNum++;
            fillWithZero( &tempPage );
        }
    }

    if (programSize % PAGESIZE){  //left over in last page
        writeSwapPage( os->secStorage, &tempPage, pageNum);

        os->pageTable[pageNum].frameNumber = pageNum;
        os->pageTable[pageNum].validBit = VALID;
        os->pageTable[pageNum].whereBit = IN_DISK;  
    }

    return 1;

}

int locatePTE( OS* os, int pageNum, int* victimPageNum, pte* pageTableEntry )
{

    //Check for out of range page number
    if ( os->pageTable[pageNum].validBit == INVALID )
        return 0;

    //Let check whether the target page is memory resident
    if ( os->pageTable[pageNum].whereBit == IN_DISK ){ //page fault!
        handlePageFault( os, pageNum, victimPageNum );
        //Page is now memory resident
    } else {
        //Memory resident page, no victim
        *victimPageNum = -1;
    }

    //Copy the whole PTE for the caller
    *pageTableEntry = os->pageTable[pageNum];
    return 1;
}

int handlePageFault( OS* os, int pageNum, int* victimPageNum )
{
    int swapPageNum, frameNumber, replacedPageNum;
    page tempPage;

    //Remember, the "frameNumber" field is used as the swap page#
    // if the page is in disk

    swapPageNum = os->pageTable[pageNum].frameNumber;

    //TODO:
    //   Currently, we always load a page into frame 0. Other frames are unused.
    //   Change this to a simple FIFO page replacement algorithm.
    //    i.e. Frame 0 is used, followed by frame 1, 2, .... 
    //           then wraps around

    //   Secondly, the replaced frame is NOT written to the
    //   corresponding swap page. Figure out how to check whether 
    //   a page is dirty and perform the corresponding write

    frameNumber = 0;

    //Check who's the page to be replaced?
    replacedPageNum = os->frameUsageTable[ frameNumber ];

    //Inform the CPU the kill off the victim page TLB entry
    *victimPageNum = replacedPageNum;

    //Update the victim's PTE
    if ( replacedPageNum != -1 ){
        os->pageTable[ replacedPageNum ].whereBit = IN_DISK;
        //In our case, the page is always written back to the same swap file page
        os->pageTable[ replacedPageNum ].frameNumber = replacedPageNum;
    }

    //Load the page from secondary storage
    readSwapPage( os->secStorage, &tempPage, swapPageNum );

    //Store the page into the chosen physical frame
    writeOneFrame( os->physicalMem, &tempPage, frameNumber );

    //Update Page Table Entry

    os->pageTable[pageNum].frameNumber = frameNumber;
    os->pageTable[pageNum].whereBit = IN_MEMORY;

    //Update Physical Frame Usage
    os->frameUsageTable[ frameNumber ] = pageNum;

    return frameNumber;

}

void printOperatingSystem( OS* os)
{
    int i;

    printf("------- Page Table --------\n");
    for ( i = 0; i < LOGICALPAGES; i++){
        printf("Page %2i| ", i);
        printPTE( &(os->pageTable[i]));
    }     


}
/**************** CPU ****************************/
void initCPU( cpu* theCPU, memory* mem, OS* os )
{
    int i;

    theCPU->operatingSystem = os;
    theCPU->physicalMem = mem;
    for(i = 0; i < TLBSIZE; i++){
        theCPU->TLB[i].enableBit = DISABLED;
    }

}

int searchTLB( tlbe* TLB, int pageNum )
{
    int result = -1, i;

    for (i = 0; (result == -1) && (i < TLBSIZE); i++){
        if (TLB[i].enableBit == ENABLED && TLB[i].pteNumber == pageNum)
            result = i;

    }

    return result;
}

int findFrameNumber(cpu* theCPU, int pageNum )
{
    int tlbeIdx, frameNum, victimPageNum, tlbeIdxForVictim;
    pte tempPTE;

    //Check TLB for page table entry
    tlbeIdx = searchTLB( theCPU->TLB, pageNum );

    if (tlbeIdx != -1){//TLB-Hit

    } else {             //TLB-Miss

        //Ask the OS about this PTE
        if(!locatePTE( theCPU->operatingSystem, 
                        pageNum, &victimPageNum, &tempPTE )){
            //OS cannot locate ==> invalid page number
            return -1;
        }
        //If any page is replaced, its corresponding TLB entry must be disabled
        if (victimPageNum != -1){
            tlbeIdxForVictim = searchTLB( theCPU->TLB, victimPageNum );
            if (tlbeIdxForVictim != -1){   //TLB entry found for victim page
                theCPU->TLB[tlbeIdxForVictim].enableBit = DISABLED;
            }
        }

        //Store the new PTE in TLB
        for (tlbeIdx=0; tlbIdx<TLBSIZE; tlbeIndx++) {
            theCPU
        }
        //TODO:
        //   Currently, the 0th entry in TLB is always replaced by new 
        //    PTE coming in. 
        //   Change this part, so that TLB to be replaced is chosen as
        //    follows:
        //     - If there is any diabled TLB entry, it is chosen
        //         - The TLB with lowest index is used when there are
        //         multiple disabled entries.
        //     - If there is no disabled TLB entry, the 0th entry is
        //     used.

        tlbeIdx = tlbeIdx % TLBSIZE;   //Choose index 0 if no free index
        theCPU->TLB[tlbeIdx].pageTableEntry = tempPTE;
        theCPU->TLB[tlbeIdx].enableBit = ENABLED;
        theCPU->TLB[tlbeIdx].pteNumber = pageNum;
    }

    //At this point the TLB entry is valid
    // and the page is memory resident
    frameNum = theCPU->TLB[tlbeIdx].pageTableEntry.frameNumber;

    return frameNum;
}

int readMemAccess( cpu* theCPU, int pageNum, int offset, int* value)
{
    int result = 0, frameNum;

    if (offset >= PAGESIZE )
        return 0;

    // The findFrameNumber performs all the important steps in 
    //  resolving memory access, see the function for more details
    frameNum = findFrameNumber( theCPU, pageNum);
    if (frameNum == -1)
        return 0;

    //If everything is handled properly up to this point, the read cannot fail
    result = readOneWord( theCPU->physicalMem, frameNum, offset, value );

    return result;
}


int writeMemAccess( cpu* theCPU, int pageNum, int offset, int value)
{
    int result = 0, frameNum;

    if (offset >= PAGESIZE )
        return 0;

    // The findFrameNumber performs all the important steps in 
    //  resolving memory access, see the function for more details
    frameNum = findFrameNumber( theCPU, pageNum);

    //Perform the write
    result = writeOneWord( theCPU->physicalMem, frameNum, offset, value );

    return result;
}

void printCPU( cpu* theCPU)
{
    int i;

    printf("---- Translation Look-Aside Buffer ----\n");

    for(i = 0; i < TLBSIZE; i++){
        printf("Entry %2i| ", i);
        if (theCPU->TLB[i].enableBit == ENABLED){
            printf(" => PTE[%2i]: ", theCPU->TLB[i].pteNumber);
            printPTE( &theCPU->TLB[i].pageTableEntry );
        } else {
            printf("INVALID\n");
        }
    }

}
