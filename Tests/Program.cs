using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.ObjectModel;
using System.Collections;
using System.IO;
using System.Reflection;
using CUDA_ML_Libary;
namespace Tests
{
    class Program
    {

        static void Main(/*string[] args*/)
        {
            #region Test CUDA_ML_Libary.MachineLearning.MachineLearningTools.GeneralTool  |CreatPolynoumsArray/CUDACreatPolynoumsArray|
            //    double[] array1 = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            //    double[][] array2 = new double[][] { array1, array1, array1, array1, array1, array1, array1, array1,};
            //    double[][][] array3 = new double[][][] { array2 , array2 , array2 , array2 , array2, array2, array2,};

            //    double[] result1;
            //    double[][] result2;
            //    double[][][] result3;

            //    double expoent = 2;
            //    double[] expoents = new double[] { 1, 2, 3 };
            //    MachineLearning.MachineLearningTools.GeneralTool.ExpoentFeatureAddress address = new MachineLearning.MachineLearningTools.GeneralTool.ExpoentFeatureAddress(2, 4);
            //    MachineLearning.MachineLearningTools.GeneralTool.ExpoentFeatureAddress[] addresses = new MachineLearning.MachineLearningTools.GeneralTool.ExpoentFeatureAddress[] 
            //    { 
            //        new MachineLearning.MachineLearningTools.GeneralTool.ExpoentFeatureAddress(2, 2),
            //        new MachineLearning.MachineLearningTools.GeneralTool.ExpoentFeatureAddress(2, 3), 
            //        new MachineLearning.MachineLearningTools.GeneralTool.ExpoentFeatureAddress(2, 4),                
            //    };
            //    bool append = true; 
            //    bool dontAppend = false;
            //    bool CUDA = true;
            //    bool isntCUDA = false;
            //    //MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(array1);

            //    ///Como funcionará a lista resultInformation:
            //    ///para cada teste será informado
            //    ///1°Argumentos informados para execuçãod a função
            //    ///2°Tempo de início e término
            //    ///3°Se é uma função utilizando CUDA ou não
            //    List<string> resultInformation = new List<string>();
            //    DateTime start, end;

            //    //Console.WriteLine(WriteArray(array1) + "\n\n\n");
            //    //Console.WriteLine(WriteArray(array2) + "\n\n\n");
            //    //Console.WriteLine(WriteArray(array3) + "\n\n\n");
            //    //Console.ReadKey();


            //    //Será feito um total de 4 teste com 12 sobrecargas em cada teste, totalizando 48 subtestes
            //    int testCount = 0;
            //    int subtestCount = 0;



            //TEST1:
            //    #region Test 1
            //    ///Teste para as seguintes sobrecargas
            //    ///public static double[] CreatPolynoumsArray(double expoent, double[] array, bool appendArray = false)
            //    ///public static double[] CUDACreatPolynoumsArray(double expoent, double[] array, bool appendArray = false)
            //    ///public static double[][] CreatPolynoumsArray(double expoent, double[][] array, bool appendArray = false)
            //    ///public static double[][] CUDACreatPolynoumsArray(double expoent, double[][] array, bool appendArray = false)
            //    ///public static double[][][] CreatPolynoumsArray(double expoent, double[][][] array, bool appendArray = false)
            //    ///public static double[][][] CUDACreatPolynoumsArray(double expoent, double[][][] array, bool appendArray = false)


            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoent, array1, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoent, array1, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoent, array1, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoent, array1, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoent, array2, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoent, array2, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoent, array2, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoent, array2, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoent, array3, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoent, array3, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoent, array3, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoent, array3, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{expoent}}}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    #endregion
            //    Console.WriteLine($"Teste {++testCount} Concluido"); Console.ReadLine();
            //TEST2:
            //    #region Test 2
            //    ///Teste para as seguintes sobrecargas
            //    ///public static double[] CreatPolynoumsArray(double[] expoent, double[] array, bool appendArray = false)                
            //    ///public static double[] CUDACreatPolynoumsArray(double[] expoent, double[] array, bool appendArray = false)                
            //    ///public static double[][] CreatPolynoumsArray(double[] expoent, double[][] array, bool appendArray = false)                
            //    ///public static double[][] CUDACreatPolynoumsArray(double[] expoent, double[][] array, bool appendArray = false)                
            //    ///public static double[][][] CreatPolynoumsArray(double[] expoent, double[][][] array, bool appendArray = false)                
            //    ///public static double[][][] CUDACreatPolynoumsArray(double[] expoent, double[][][] array, bool appendArray = false)


            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoents, array1, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoents, array1, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoents, array1, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoents, array1, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoents, array2, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoents, array2, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoents, array2, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoents, array2, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoents, array3, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoents, array3, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(expoents, array3, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(expoents, array3, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {{{WriteArray(expoents)}}}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    #endregion
            //    Console.WriteLine($"Teste {++testCount} Concluido"); Console.ReadLine();
            //TEST3:
            //    #region Test 3
            //    ///Teste para as seguintes sobrecargas
            //    ///public static double[] CreatPolynoumsArray(ExpoentFeatureAddress expoent, double[] array, bool appendArray = false)                
            //    ///public static double[] CUDACreatPolynoumsArray(ExpoentFeatureAddress expoent, double[] array, bool appendArray = false)                
            //    ///public static double[][] CreatPolynoumsArray(ExpoentFeatureAddress expoent, double[][] array, bool appendArray = false)                
            //    ///public static double[][] CUDACreatPolynoumsArray(ExpoentFeatureAddress expoent, double[][] array, bool appendArray = false)                
            //    ///public static double[][][] CreatPolynoumsArray(ExpoentFeatureAddress expoent, double[][][] array, bool appendArray = false)                
            //    ///public static double[][][] CUDACreatPolynoumsArray(ExpoentFeatureAddress expoent, double[][][] array, bool appendArray = false)


            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(address, array1, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(address, array1, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(address, array1, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(address, array1, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(address, array2, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(address, array2, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(address, array2, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(address, array2, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(address, array3, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(address, array3, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(address, array3, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(address, array3, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {address}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    #endregion
            //    Console.WriteLine($"Teste {++testCount} Concluido"); Console.ReadLine();            
            //TEST4:
            //    #region Test 4
            //    ///Teste para as seguintes sobrecargas
            //    ///public static double[] CreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[] array, bool appendArray = false)                
            //    ///public static double[] CUDACreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[] array, bool appendArray = false)                
            //    ///public static double[][] CreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[][] array, bool appendArray = false)                
            //    ///public static double[][] CUDACreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[][] array, bool appendArray = false)                
            //    ///public static double[][][] CreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[][][] array, bool appendArray = false)                
            //    ///public static double[][][] CUDACreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[][][] array, bool appendArray = false)


            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(addresses, array1, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(addresses, array1, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(addresses, array1, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result1 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(addresses, array1, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array1)}\n{{WriteArray(array1)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result1)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(addresses, array2, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(addresses, array2, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(addresses, array2, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result2 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(addresses, array2, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array2)}\n{{WriteArray(array2)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result2)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(addresses, array3, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {dontAppend}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(addresses, array3, dontAppend);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {dontAppend}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");

            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CreatPolynoumsArray(addresses, array3, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {append}; \nCUDA:{isntCUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    start = DateTime.Now;
            //    result3 = MachineLearning.MachineLearningTools.GeneralTool.CUDACreatPolynoumsArray(addresses, array3, append);
            //    end = DateTime.Now;
            //    /*Console.WriteLine(++subtestCount);Console.ReadKey();*/resultInformation.Add($"Result test with the following arguments: \nExpoent: {WriteArray(addresses)}; \nArray Name: {nameof(array3)}\n{{WriteArray(array3)}}\n \nAppend Array: {append}; \nCUDA:{CUDA}. \nStart: {start}\nEnd: {end} \nTotal: {end.Subtract(start)}\n\n Result:\n{{WriteArray(result3)}}\n\n\n\n\n\n\n\n");
            //    #endregion
            //    Console.WriteLine($"Teste {++testCount} Concluido"); Console.ReadLine();
            //FINAL:
            //    Console.WriteLine("Concluido");
            //    Console.WriteLine(string.Concat(resultInformation));
            //    Console.ReadLine();
            //    return;

            #endregion

            #region Test CUDA_ML_Libary.MachineLearning.MachineLearningTools.Regularization  |ScaleAdjust/CUDAScaleAdjust|    |MeanNormalization/CUDAMeanNormalization|
            //    double[] array0 = new double[]    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            //    double[] array1 = new double[]    { 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 };
            //    double[] array10 = new double[]   { 3, 4, 5, 6, 7, 8, 9, 10, 1, 2 };
            //    double[] array11 = new double[]   { 4, 5, 6, 7, 8, 9, 10, 1, 2, 3 };
            //    double[] array100 = new double[]  { 5, 6, 7, 8, 9, 10, 1, 2, 3, 4 };
            //    double[] array101 = new double[]  { 6, 7, 8, 9, 10, 1, 2, 3, 4, 5 };
            //    double[] array110 = new double[]  { 7, 8, 9, 10, 1, 2, 3, 4, 5, 6 };
            //    double[] array111 = new double[]  { 8, 9, 10, 1, 2, 3, 4, 5, 6, 7 };
            //    double[] array1000 = new double[] { 9, 10, 1, 2, 3, 4, 5, 6, 7, 8 };
            //    double[] array1001 = new double[] { 10, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            //    double[][] array2 = new double[][] { array0, array1, array10, array11, array100, array101, array110, array111, array1000, array1001, };     

            //    double[][][] array3 = new double[][][] { array2, array2, array2, array2 };

            //    double[] result1;
            //    double[][] result2;
            //    double[][][] result3;



            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.ScaleAdjust(array0)));
            //    Console.WriteLine();
            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.CUDAScaleAdjust(array0)));
            //    Console.WriteLine();

            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.MeanNormalization(array0)));
            //    Console.WriteLine();
            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.CUDAMeanNormalization(array0)));
            //    Console.WriteLine();

            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.MeanNormalization(MachineLearning.MachineLearningTools.Regularization.ScaleAdjust(array0))));
            //    Console.WriteLine();
            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.CUDAMeanNormalization(MachineLearning.MachineLearningTools.Regularization.CUDAScaleAdjust(array0))));
            //    Console.WriteLine();


            //    //MachineLearning.MachineLearningTools.Regularization.ScaleAdjust(ref array0);
            //    //Console.WriteLine(WriteArray(array0));
            //    //Console.ReadKey();
            //    //MachineLearning.MachineLearningTools.Regularization.CUDAScaleAdjust(ref array0);
            //    //Console.WriteLine(WriteArray(array0));
            //    //Console.ReadKey();

            //    //MachineLearning.MachineLearningTools.Regularization.MeanNormalization(ref array0);
            //    //Console.WriteLine(WriteArray(array0));
            //    //Console.ReadKey();
            //    //MachineLearning.MachineLearningTools.Regularization.CUDAMeanNormalization(ref array0);
            //    //Console.WriteLine(WriteArray(array0));
            //    //Console.ReadKey();




            //    Console.WriteLine("-------------------------------------------------------------------------------------------------------");
            //    Console.WriteLine(WriteArray(array2));
            //    Console.ReadKey();
            //    Console.WriteLine("-------------------------------------------------------------------------------------------------------");

            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.ScaleAdjust(array2)));
            //    Console.WriteLine();
            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.CUDAScaleAdjust(array2)));
            //    Console.WriteLine();

            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.MeanNormalization(array2)));
            //    Console.WriteLine();
            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.CUDAMeanNormalization(array2)));
            //    Console.WriteLine();

            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.MeanNormalization(MachineLearning.MachineLearningTools.Regularization.ScaleAdjust(array2))));
            //    Console.WriteLine();
            //    Console.WriteLine(WriteArray(MachineLearning.MachineLearningTools.Regularization.CUDAMeanNormalization(MachineLearning.MachineLearningTools.Regularization.CUDAScaleAdjust(array2))));
            //    Console.WriteLine();


            //FINAL:
            //    Console.WriteLine("Concluido");
            //    Console.ReadLine();
            //    return;

            #endregion

            #region Test CUDA_ML_Libary.MachineLearning.MachineLearningTools.Regularization  |DoBatchGradientDescent|    |CUDADoBatchGradientDescent|
            //double[] array0 = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            //double[] array1 = new double[] { 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 };
            //double[] array10 = new double[] { 3, 4, 5, 6, 7, 8, 9, 10, 1, 2 };
            //double[] array11 = new double[] { 4, 5, 6, 7, 8, 9, 10, 1, 2, 3 };
            //double[] array100 = new double[] { 5, 6, 7, 8, 9, 10, 1, 2, 3, 4 };
            //double[] array101 = new double[] { 6, 7, 8, 9, 10, 1, 2, 3, 4, 5 };
            //double[] array110 = new double[] { 7, 8, 9, 10, 1, 2, 3, 4, 5, 6 };
            //double[] array111 = new double[] { 8, 9, 10, 1, 2, 3, 4, 5, 6, 7 };
            //double[] array1000 = new double[] { 9, 10, 1, 2, 3, 4, 5, 6, 7, 8 };
            //double[] array1001 = new double[] { 10, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            //double[][] array2 = new double[][] { array0, array1, array10, array11, array100, array101, array110, array111, array1000, array1001, };

            //double[][][] array3 = new double[][][] { array2, array2, array2, array2 };

            //double[] result1;
            //double[][] result2;
            //double[][][] result3;
            #endregion

            #region Test CUDA_ML_Libary.MachineLearning.SupervisedLearning.LinearRegression  |DoBatchGradientDescent|    |CUDADoBatchGradientDescent|
            //    string path = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Relatório.txt";


            //    IEnumerable<double>
            //        in_x1 = new double[] { },//Quantidade de entradas
            //        out_x2 = new double[] { },//Quantidade de saídas
            //        highOut_x3 = new double[] { },//Quantidade de saídas com nível lógico alto
            //        expQntd_x4 = new double[] { },//Tamanho da expressão (s/ simplificação)
            //        simpExpQntd_x5 = new double[] { };//Tamanho da expressão (c/ simplificação)
            //    IEnumerable<double>
            //        ttl_y1 = new double[] { };

            //    Datas datas = new Datas();

            //    List<double[]> features = new List<double[]>();

            //    foreach (string line in File.ReadLines(path))
            //    {
            //        if (line.Contains("Inputs;Outputs;Expressions Count;LogicalOperatorExpression Lenght CSHARP;Simplify Start;Simplify End; End-Start;Simplified Expression Lenght CSHARP"))
            //            continue;
            //        string[] data = line.Split(';');

            //        double
            //        x1 = Convert.ToDouble(data[0]),
            //        x2 = Convert.ToDouble(data[1]),
            //        x3 = Convert.ToDouble(data[2]),
            //        x4 = Convert.ToDouble(data[3]),
            //        x5 = Convert.ToDouble(data[7]);
            //        double
            //        y1 = TimeSpan.Parse(data[6]).Ticks;

            //        datas.Add
            //            (
            //                new Data
            //                (
            //                    Convert.ToInt32(data[0]),
            //                    Convert.ToInt32(data[1]),
            //                    Convert.ToInt32(data[2]),
            //                    Convert.ToInt32(data[3]),
            //                    DateTime.Parse(data[4]),
            //                    DateTime.Parse(data[5]),
            //                    TimeSpan.Parse(data[6]),
            //                    Convert.ToInt32(data[7])
            //                )
            //             );

            //        in_x1 = in_x1.Append(x1);
            //        out_x2 = out_x2.Append(x2);
            //        highOut_x3 = highOut_x3.Append(x3);
            //        expQntd_x4 = expQntd_x4.Append(x4);
            //        simpExpQntd_x5 = simpExpQntd_x5.Append(x5);
            //        ttl_y1 = ttl_y1.Append(y1);

            //        features.Add(new double[] { x1, x2, x3, x4, x5 });
            //    }

            //    //double[][] features = new double[][]
            //    //{
            //    //    in_x1.ToArray(),
            //    //    out_x2.ToArray(),
            //    //    highOut_x3.ToArray(),
            //    //    expQntd_x4.ToArray(),
            //    //    simpExpQntd_x5.ToArray(),
            //    //};

            //    double[] output = ttl_y1.ToArray();

            //    DateTime a = DateTime.Now, b = DateTime.Now, c = DateTime.Now, d = DateTime.Now;
            //    a = DateTime.Now;
            //    MachineLearning.MLDatas mlDatas = new MachineLearning.MLDatas(features.ToArray(), output, 60, 20, 20);
            //    b = DateTime.Now;
            //    //c = DateTime.Now;
            //    //MachineLearning.MLDatas cudamlDatas = new MachineLearning.CUDAMLDatas(features.ToArray(), output, 60, 20, 20);
            //    //d = DateTime.Now;

            //    MachineLearning.SupervisedLearning.LinearRegression lR = new MachineLearning.SupervisedLearning.LinearRegression(mlDatas, 0.00000000005, 50, true);

            //    #region Batch 
            //    //Console.WriteLine(WriteArray(lR.Theta));
            //    //Console.WriteLine();
            //    //a = DateTime.Now;
            //    //lR.DoBatchGradientDescent(20000);
            //    //b = DateTime.Now;
            //    //Console.WriteLine(WriteArray(lR.Theta));
            //    //Console.WriteLine();
            //    //Console.ReadLine();
            //    //for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
            //    //{
            //    //    double predict = lR.Theta[0];
            //    //    for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
            //    //        predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

            //    //    Console.WriteLine($"{predict} / {lR.Datas.TestOutputs[m]} = {predict / lR.Datas.TestOutputs[m] * 100}%");
            //    //}

            //    //c = DateTime.Now;
            //    //lR.CUDADoBatchGradientDescent(20000);
            //    //d = DateTime.Now;
            //    //Console.WriteLine(WriteArray(lR.Theta));
            //    //Console.WriteLine();
            //    //Console.ReadLine();
            //    //for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
            //    //{
            //    //    double predict = lR.Theta[0];
            //    //    for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
            //    //        predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

            //    //    Console.WriteLine($"{predict} / {lR.Datas.TestOutputs[m]} = {predict / lR.Datas.TestOutputs[m] * 100}%");
            //    //}
            //    #endregion
            //    #region Stochastic
            //    //Console.WriteLine(WriteArray(lR.Theta));
            //    //Console.WriteLine();
            //    //a = DateTime.Now;
            //    //lR.DoStochasticGradientDescent(200000);
            //    //b = DateTime.Now;
            //    //Console.WriteLine(WriteArray(lR.Theta));
            //    //Console.WriteLine();
            //    //Console.ReadLine();
            //    //for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
            //    //{
            //    //    double predict = lR.Theta[0];
            //    //    for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
            //    //        predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

            //    //    Console.WriteLine($"{predict} / {lR.Datas.TestOutputs[m]} = {predict / lR.Datas.TestOutputs[m] * 100}%");
            //    //}
            //    //c = DateTime.Now;
            //    //lR.CUDADoStochasticGradientDescent(200000);
            //    //d = DateTime.Now;
            //    //Console.WriteLine(WriteArray(lR.Theta));
            //    //Console.WriteLine();
            //    //Console.ReadLine();
            //    //for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
            //    //{
            //    //    double predict = lR.Theta[0];
            //    //    for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
            //    //        predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

            //    //    Console.WriteLine($"{predict} / {lR.Datas.TestOutputs[m]} = {predict / lR.Datas.TestOutputs[m] * 100}%");
            //    //}
            //    #endregion

            //    Console.Write
            //        (
            //        $"Start: {a}\n" +
            //        $"End: {b}\n" +
            //        $"Total: {b.Subtract(a)}\n" +
            //        $"Start: {c}\n" +
            //        $"End: {d}\n" +
            //        $"Total: {d.Subtract(c)}\n"
            //        );
            //FINAL:
            //    Console.WriteLine("Concluido");
            //    Console.ReadLine();
            //    return;
            #endregion

            #region Test CUDA_ML_Libary.MachineLearning.SupervisedLearning.LinearRegression  |DoBatchGradientDescent|    |CUDADoBatchGradientDescent| (pt2)
            //    string path = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Linear Regression\LR_DataSample.txt";

            //    IEnumerable<double>
            //        f1_x1 = new double[] { },
            //        f2_x2 = new double[] { },
            //        f3_x3 = new double[] { },
            //        f4_x4 = new double[] { },
            //        f5_x5 = new double[] { };
            //    IEnumerable<double>
            //        out_y1 = new double[] { };

            //    List<double[]> features = new List<double[]>();

            //    foreach (string line in File.ReadLines(path))
            //    {
            //        if (line.Contains("Inputs;Outputs;Expressions Count;LogicalOperatorExpression Lenght CSHARP;Simplify Start;Simplify End; End-Start;Simplified Expression Lenght CSHARP"))
            //            continue;
            //        string[] data = line.Split(',');

            //        double
            //        x1 = Convert.ToDouble(data[0]),
            //        x2 = Convert.ToDouble(data[1]),
            //        x3 = Convert.ToDouble(data[2]),
            //        x4 = Convert.ToDouble(data[3]),
            //        x5 = Convert.ToDouble(data[4]);
            //        double
            //        y1 = Convert.ToDouble(data[5]);


            //        f1_x1 = f1_x1.Append(x1);
            //        f2_x2 = f2_x2.Append(x2);
            //        f3_x3 = f3_x3.Append(x3);
            //        f4_x4 = f4_x4.Append(x4);
            //        f5_x5 = f5_x5.Append(x5);
            //        out_y1 = out_y1.Append(y1);

            //        features.Add(new double[] { x1, x2, x3, x4, x5 });
            //    }

            //    //double[][] features = new double[][]
            //    //{
            //    //    in_x1.ToArray(),
            //    //    out_x2.ToArray(),
            //    //    highOut_x3.ToArray(),
            //    //    expQntd_x4.ToArray(),
            //    //    simpExpQntd_x5.ToArray(),
            //    //};

            //    double[] output = out_y1.ToArray();

            //    DateTime a = DateTime.Now, b = DateTime.Now, c = DateTime.Now, d = DateTime.Now;
            //    a = DateTime.Now;
            //    MachineLearning.MLDatas mlDatas = new MachineLearning.MLDatas(features.ToArray(), output, 60, 20, 20);
            //    b = DateTime.Now;
            //    //c = DateTime.Now;
            //    //MachineLearning.MLDatas cudamlDatas = new MachineLearning.CUDAMLDatas(features.ToArray(), output, 60, 20, 20);
            //    //d = DateTime.Now;

            //    MachineLearning.SupervisedLearning.LinearRegression lR = new MachineLearning.SupervisedLearning.LinearRegression(mlDatas, 0.0000000051499, 1000000, true);

            //    #region Batch 
            //    Console.WriteLine(WriteArray(lR.Theta));
            //    Console.WriteLine();

            //    a = DateTime.Now;
            //    lR.DoBatchGradientDescent(1000, true, MachineLearning.SupervisedLearning.LinearRegression.Normalization.MaxMin_Normalization, true/*, false, false, false*/);
            //    b = DateTime.Now;
            //    List<double> percentages = new List<double>();
            //    double max = double.NaN, min = double.NaN;
            //    for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
            //    {
            //        double predict = lR.Theta[0];
            //        for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
            //            predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];
            //        Console.WriteLine($"{predict} / {lR.Datas.TestOutputs[m][0]} = {(lR.Datas.TestOutputs[m][0] != 0 ? predict / lR.Datas.TestOutputs[m][0] * 100 : 0)}%");
            //        percentages.Add((lR.Datas.TestOutputs[m][0] != 0 ? predict / lR.Datas.TestOutputs[m][0] * 100 : 0));
            //        if (double.IsNaN(max) || max < percentages.Last())
            //            max = percentages.Last();
            //        if (double.IsNaN(min) || min > percentages.Last())
            //            min = percentages.Last();
            //    }
            //    Console.WriteLine($"Thetas: {{{string.Join(";", lR.Theta)}}}");
            //    Console.WriteLine($"Average: {percentages.Average()}%        Standart Deviation: {GetStandartDeviation(percentages)}%");
            //    Console.WriteLine($"Max: {max}%        Min: {min}%");
            //    Console.ReadLine();
            //    c = DateTime.Now;
            //    lR.CUDADoBatchGradientDescent(1000, true, MachineLearning.SupervisedLearning.LinearRegression.Normalization.MaxMin_Normalization, true/*, false, false, false*/);
            //    d = DateTime.Now;

            //    percentages.Clear();
            //    max = double.NaN; min = double.NaN;
            //    for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
            //    {
            //        double predict = lR.Theta[0];
            //        for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
            //            predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

            //        Console.WriteLine($"{predict} / {lR.Datas.TestOutputs[m][0]} = {(lR.Datas.TestOutputs[m][0] != 0 ? predict / lR.Datas.TestOutputs[m][0] * 100 : 0)}%");
            //        percentages.Add((lR.Datas.TestOutputs[m][0] != 0 ? predict / lR.Datas.TestOutputs[m][0] * 100 : 0));
            //        if (double.IsNaN(max) || max < percentages.Last())
            //            max = percentages.Last();
            //        if (double.IsNaN(min) || min > percentages.Last())
            //            min = percentages.Last();
            //    }
            //    Console.WriteLine($"Thetas: {{{string.Join(";", lR.Theta)}}}");
            //    Console.WriteLine($"Average: {percentages.Average()}%        Standart Deviation: {GetStandartDeviation(percentages)}%");
            //    Console.WriteLine($"Max: {max}%        Min: {min}%");
            //    Console.ReadLine();
            //    #endregion
            //    #region Stochastic

            //    #endregion
            //    #region Mini-Batch

            //    #endregion
            //    Console.Write
            //        (
            //        $"Start: {a}\n" +
            //        $"End: {b}\n" +
            //        $"Total: {b.Subtract(a)}\n" +
            //        $"Start: {c}\n" +
            //        $"End: {d}\n" +
            //        $"Total: {d.Subtract(c)}\n"
            //        );
            //FINAL:
            //    Console.WriteLine("Concluido");
            //    Console.ReadLine();
            //    return;
            #endregion

            #region Test CUDA_ML_Libary.MachineLearning.SupervisedLearning.LogisticRegression  |DoBatchGradientDescent|    |CUDADoBatchGradientDescent|
            //    string path = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Logistic Regression\LR_DataSample.txt";

            //    IEnumerable<double>
            //        f1_x1 = new double[] { },//Quantidade de entradas
            //        f2_x2 = new double[] { };//Quantidade de saídas                
            //    IEnumerable<double>
            //        out_y1 = new double[] { };

            //    List<double[]> features = new List<double[]>();

            //    foreach (string line in File.ReadLines(path))
            //    {
            //        if (line.Contains("Inputs;Outputs;Expressions Count;LogicalOperatorExpression Lenght CSHARP;Simplify Start;Simplify End; End-Start;Simplified Expression Lenght CSHARP"))
            //            continue;
            //        string[] data = line.Split(',');

            //        double
            //        x1 = Convert.ToDouble(data[0].Replace('.', ',')),
            //        x2 = Convert.ToDouble(data[1].Replace('.', ','));
            //        double
            //        y1 = Convert.ToDouble(data[2]);


            //        f1_x1 = f1_x1.Append(x1);
            //        f2_x2 = f2_x2.Append(x2);
            //        out_y1 = out_y1.Append(y1);

            //        features.Add(new double[] { x1, x2 });
            //    }

            //    //double[][] features = new double[][]
            //    //{
            //    //    in_x1.ToArray(),
            //    //    out_x2.ToArray(),
            //    //    highOut_x3.ToArray(),
            //    //    expQntd_x4.ToArray(),
            //    //    simpExpQntd_x5.ToArray(),
            //    //};

            //    double[] output = out_y1.ToArray();

            //    DateTime a = DateTime.Now, b = DateTime.Now, c = DateTime.Now, d = DateTime.Now;
            //    a = DateTime.Now;
            //    MachineLearning.MLDatas mlDatas = new MachineLearning.MLDatas(features.ToArray(), output, 60, 20, 20);
            //    b = DateTime.Now;
            //    //c = DateTime.Now;
            //    //MachineLearning.MLDatas cudamlDatas = new MachineLearning.CUDAMLDatas(features.ToArray(), output, 60, 20, 20);
            //    //d = DateTime.Now;

            //    MachineLearning.SupervisedLearning.LogisticRegression lR = new MachineLearning.SupervisedLearning.LogisticRegression(mlDatas, 0.00005, 20, true);

            //    #region Batch 
            //    Console.WriteLine(WriteArray(lR.Theta));
            //    Console.WriteLine();

            //    a = DateTime.Now;
            //    lR.DoBatchGradientDescent(50, false, false, false);
            //    b = DateTime.Now;
            //    double
            //        tp,// true positive
            //        tn,// true negative
            //        fp,// false positive 
            //        fn;// false negative

            //    double
            //        acc,//precisão (accuracy) mede a proporção de predições corretas, independemente de serem verdadeiros positivos ou negativos.
            //        sens,//sensibilidade (sensitivity) mede a proporção de verdadeiros positivos, ou seja, a capacidade do modelo classificar como positivo dado que ele é de fato positivo.
            //        spe;//especificidade (pecificity) mede a proporção de verdadeiros negativos, ou seja, a capacidade do modelo classificar como negativo dado que ele é de fato negativo.

            //    tp = 0; tn = 0; fp = 0; fn = 0;
            //    for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
            //    {
            //        double predict = lR.Theta[0];
            //        for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
            //            predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

            //        int
            //            pred = ((predict) >= 0.5 ? 1 : 0),
            //            res = (pred == lR.Datas.TestOutputs[m] ? 100 : 0);
            //        if (res == 100)
            //        {
            //            if (pred == 1)
            //                tp++;
            //            else
            //                tn++;
            //        }
            //        else
            //        {
            //            if (pred == 1)
            //                fn++;
            //            else
            //                fp++;
            //        }

            //        Console.WriteLine($"{{{predict}}} -> {((predict) >= 0.5 ? 1 : 0)} / {lR.Datas.TestOutputs[m]} = {res}%");
            //    }

            //    //EXEMPLO DE UMA CONFIGURAÇÃO DE FORNECE UMA PRECISÃO ALTA
            //    //Thetas: {-0,000111772911285156;0,00443780580073447;0,00416591991433254}
            //    //tp: 15    tn: 4    fp: 0    fn: 1
            //    //acc: 95 % sens: 93,75 % spe: 100 %
            //    Console.WriteLine($"Thetas: {{{string.Join(";", lR.Theta)}}}");
            //    Console.WriteLine($"tp: {tp}    tn: {tn}    fp: {fp}    fn: {fn}");
            //    acc = (tp + tn) / lR.Datas.TestFeatures.Length; //precisão (accuracy) mede a proporção de predições corretas, independemente de serem verdadeiros positivos ou negativos.
            //    sens = tp / (tp + fn);//sensibilidade (sensitivity) mede a proporção de verdadeiros positivos, ou seja, a capacidade do modelo classificar como positivo dado que ele é de fato positivo.
            //    spe = tn / (tn + fp);//especificidade (pecificity) mede a proporção de verdadeiros negativos, ou seja, a capacidade do modelo classificar como negativo dado que ele é de fato negativo.
            //    Console.WriteLine($"acc: {acc * 100}%    sens: {sens * 100}%    spe: {spe * 100}%");
            //    Console.ReadLine();
            //    c = DateTime.Now;
            //    lR.CUDADoBatchGradientDescent(50/*, false, false, false*/);
            //    d = DateTime.Now;
            //    Console.WriteLine(WriteArray(lR.Theta));
            //    Console.WriteLine();
            //    Console.ReadLine();
            //    tp = 0; tn = 0; fp = 0; fn = 0;
            //    for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
            //    {
            //        double predict = lR.Theta[0];
            //        for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
            //            predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

            //        int
            //            pred = ((predict) >= 0.5 ? 1 : 0),
            //            res = (pred == lR.Datas.TestOutputs[m] ? 100 : 0);
            //        if (res == 100)
            //        {
            //            if (pred == 1)
            //                tp++;
            //            else
            //                tn++;
            //        }
            //        else
            //        {
            //            if (pred == 1)
            //                fn++;
            //            else
            //                fp++;
            //        }
            //        Console.WriteLine($"{{{predict}}} -> {((predict) >= 0.5 ? 1 : 0)} / {lR.Datas.TestOutputs[m]} = {res}%");
            //    }
            //    Console.WriteLine($"tp: {tp}    tn: {tn}    fp: {fp}    fn: {fn}");
            //    acc = (tp + tn) / lR.Datas.TestFeatures.Length; //precisão (accuracy) mede a proporção de predições corretas, independemente de serem verdadeiros positivos ou negativos.
            //    sens = tp / (tp + fn);//sensibilidade (sensitivity) mede a proporção de verdadeiros positivos, ou seja, a capacidade do modelo classificar como positivo dado que ele é de fato positivo.
            //    spe = tn / (tn + fp);//especificidade (pecificity) mede a proporção de verdadeiros negativos, ou seja, a capacidade do modelo classificar como negativo dado que ele é de fato negativo.
            //    Console.WriteLine($"acc: {acc * 100}%    sens: {sens * 100}%    spe: {spe * 100}%");
            //    #endregion
            //    #region Stochastic

            //    #endregion
            //    #region Mini-Batch

            //    #endregion
            //    Console.Write
            //        (
            //        $"Start: {a}\n" +
            //        $"End: {b}\n" +
            //        $"Total: {b.Subtract(a)}\n" +
            //        $"Start: {c}\n" +
            //        $"End: {d}\n" +
            //        $"Total: {d.Subtract(c)}\n"
            //        );
            //FINAL:
            //    Console.WriteLine("Concluido");
            //    Console.ReadLine();
            //    return;
            #endregion

            #region Test CUDA_ML_Libary.MachineLearning.SupervisedLearning.LogisticRegression  |BatchGradientDescent|    |CUDABatchGradientDescent|
        //    string path = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Logistic Regression\LR_DataSample.txt";

        //    IEnumerable<double>
        //        f1_x1 = new double[] { },//Quantidade de entradas
        //        f2_x2 = new double[] { };//Quantidade de saídas                
        //    IEnumerable<double>
        //        out_y1 = new double[] { };

        //    List<double[]> features = new List<double[]>();

        //    foreach (string line in File.ReadLines(path))
        //    {
        //        string[] data = line.Split(',');

        //        double
        //        x1 = Convert.ToDouble(data[0].Replace('.', ',')),
        //        x2 = Convert.ToDouble(data[1].Replace('.', ','));
        //        double
        //        y1 = Convert.ToDouble(data[2]);


        //        f1_x1 = f1_x1.Append(x1);
        //        f2_x2 = f2_x2.Append(x2);
        //        out_y1 = out_y1.Append(y1);

        //        features.Add(new double[] { x1, x2 });
        //    }

        //    //double[][] features = new double[][]
        //    //{
        //    //    in_x1.ToArray(),
        //    //    out_x2.ToArray(),
        //    //    highOut_x3.ToArray(),
        //    //    expQntd_x4.ToArray(),
        //    //    simpExpQntd_x5.ToArray(),
        //    //};

        //    double[] output = out_y1.ToArray();

        //    DateTime a = DateTime.Now, b = DateTime.Now, c = DateTime.Now, d = DateTime.Now;
        //    a = DateTime.Now;
        //    MachineLearning.MLDatas mlDatas = new MachineLearning.MLDatas(features.ToArray(), output, 60, 20, 20);
        //    b = DateTime.Now;
        //    //c = DateTime.Now;
        //    //MachineLearning.MLDatas cudamlDatas = new MachineLearning.CUDAMLDatas(features.ToArray(), output, 60, 20, 20);
        //    //d = DateTime.Now;

        //    MachineLearning.SupervisedLearning.LogisticRegression lR = new MachineLearning.SupervisedLearning.LogisticRegression(mlDatas, 0.15, 1, true);

        //    #region Batch 
        //    Console.WriteLine(WriteArray(lR.Theta));
        //    Console.WriteLine();

        //    a = DateTime.Now;
        //    lR.BatchGradientDescent(lR.Datas.TrainFeatures, MachineLearning.MachineLearningTools.GeneralTools.CUDAGetSpecificElements(0, lR.Datas.TrainOutputs), lR.Alpha, lR.Lambda, 200);
        //    //lR.DoBatchGradientDescent(200);
        //    b = DateTime.Now;
        //    double
        //        tp,// true positive
        //        tn,// true negative
        //        fp,// false positive 
        //        fn;// false negative

        //    double
        //        acc,//precisão (accuracy) mede a proporção de predições corretas, independemente de serem verdadeiros positivos ou negativos.
        //        sens,//sensibilidade (sensitivity) mede a proporção de verdadeiros positivos, ou seja, a capacidade do modelo classificar como positivo dado que ele é de fato positivo.
        //        spe;//especificidade (pecificity) mede a proporção de verdadeiros negativos, ou seja, a capacidade do modelo classificar como negativo dado que ele é de fato negativo.

        //    tp = 0; tn = 0; fp = 0; fn = 0;
        //    for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
        //    {
        //        double predict = lR.Theta[0];
        //        for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
        //            predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

        //        int
        //            pred = ((predict) >= 0.5 ? 1 : 0),
        //            res = (pred == lR.Datas.TestOutputs[m][0] ? 100 : 0);
        //        if (res == 100)
        //        {
        //            if (pred == 1)
        //                tp++;
        //            else
        //                tn++;
        //        }
        //        else
        //        {
        //            if (pred == 1)
        //                fn++;
        //            else
        //                fp++;
        //        }

        //        Console.WriteLine($"{{{predict}}} -> {((predict) >= 0.5 ? 1 : 0)} / {lR.Datas.TestOutputs[m][0]} = {res}%");
        //    }

        //    //EXEMPLO DE UMA CONFIGURAÇÃO DE FORNECE UMA PRECISÃO ALTA
        //    //Thetas: {-0,000111772911285156;0,00443780580073447;0,00416591991433254}
        //    //tp: 15    tn: 4    fp: 0    fn: 1
        //    //acc: 95 % sens: 93,75 % spe: 100 %
        //    Console.WriteLine($"Thetas: {{{string.Join(";", lR.Theta)}}}");
        //    Console.WriteLine($"tp: {tp}    tn: {tn}    fp: {fp}    fn: {fn}");
        //    acc = (tp + tn) / lR.Datas.TestFeatures.Length; //precisão (accuracy) mede a proporção de predições corretas, independemente de serem verdadeiros positivos ou negativos.
        //    sens = tp / (tp + fn);//sensibilidade (sensitivity) mede a proporção de verdadeiros positivos, ou seja, a capacidade do modelo classificar como positivo dado que ele é de fato positivo.
        //    spe = tn / (tn + fp);//especificidade (pecificity) mede a proporção de verdadeiros negativos, ou seja, a capacidade do modelo classificar como negativo dado que ele é de fato negativo.
        //    Console.WriteLine($"acc: {acc * 100}%    sens: {sens * 100}%    spe: {spe * 100}%");
        //    Console.ReadLine();
        //    c = DateTime.Now;
        //    lR.CUDABatchGradientDescent(lR.Datas.TrainFeatures, MachineLearning.MachineLearningTools.GeneralTools.CUDAGetSpecificElements(0, lR.Datas.TrainOutputs), lR.Alpha, lR.Lambda, 200, true, MachineLearning.SupervisedLearning.LogisticRegression.Normalization.Standardisation);
        //    //lR.CUDADoBatchGradientDescent(200);
        //    d = DateTime.Now;            
        //    Console.WriteLine();
        //    tp = 0; tn = 0; fp = 0; fn = 0;
        //    for (int m = 0; m < lR.Datas.TestFeatures.Length; m++)
        //    {
        //        double predict = lR.Theta[0];
        //        for (int n = 0; n < lR.Datas.TestFeatures[m].Length; n++)
        //            predict += lR.Theta[n + 1] * lR.Datas.TestFeatures[m][n];

        //        int
        //            pred = ((predict) >= 0.5 ? 1 : 0),
        //            res = (pred == lR.Datas.TestOutputs[m][0] ? 100 : 0);
        //        if (res == 100)
        //        {
        //            if (pred == 1)
        //                tp++;
        //            else
        //                tn++;
        //        }
        //        else
        //        {
        //            if (pred == 1)
        //                fn++;
        //            else
        //                fp++;
        //        }
        //        Console.WriteLine($"{{{predict}}} -> {((predict) >= 0.5 ? 1 : 0)} / {lR.Datas.TestOutputs[m][0]} = {res}%");
        //    }
        //    Console.WriteLine($"Thetas: {{{string.Join(";", lR.Theta)}}}");
        //    Console.WriteLine($"tp: {tp}    tn: {tn}    fp: {fp}    fn: {fn}");
        //    acc = (tp + tn) / lR.Datas.TestFeatures.Length; //precisão (accuracy) mede a proporção de predições corretas, independemente de serem verdadeiros positivos ou negativos.
        //    sens = tp / (tp + fn);//sensibilidade (sensitivity) mede a proporção de verdadeiros positivos, ou seja, a capacidade do modelo classificar como positivo dado que ele é de fato positivo.
        //    spe = tn / (tn + fp);//especificidade (pecificity) mede a proporção de verdadeiros negativos, ou seja, a capacidade do modelo classificar como negativo dado que ele é de fato negativo.
        //    Console.WriteLine($"acc: {acc * 100}%    sens: {sens * 100}%    spe: {spe * 100}%");
        //    #endregion
        //    #region Stochastic

        //    #endregion
        //    #region Mini-Batch

        //    #endregion
        //    Console.Write
        //        (
        //        $"Start: {a}\n" +
        //        $"End: {b}\n" +
        //        $"Total: {b.Subtract(a)}\n" +
        //        $"Start: {c}\n" +
        //        $"End: {d}\n" +
        //        $"Total: {d.Subtract(c)}\n"
        //        );
        //FINAL:
        //    Console.WriteLine("Concluido");
        //    Console.ReadLine();
        //    return;
            #endregion

            

            #region Simplifer Expressions Data
            //string path = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Relatório.txt";


            //IEnumerable<double>
            //    in_x1 = new double[] { },//Quantidade de entradas
            //    out_x2 = new double[] { },//Quantidade de saídas
            //    highOut_x3 = new double[] { },//Quantidade de saídas com nível lógico alto
            //    expQntd_x4 = new double[] { },//Tamanho da expressão (s/ simplificação)
            //    simpExpQntd_x5 = new double[] { };//Tamanho da expressão (c/ simplificação)
            //IEnumerable<double>
            //    ttl_y1 = new double[] { };

            //Datas datas = new Datas();

            //List<double[]> features = new List<double[]>();

            //foreach (string line in File.ReadAllLines(path))
            //{
            //    if (line.Contains("Inputs;Outputs;Expressions Count;LogicalOperatorExpression Lenght CSHARP;Simplify Start;Simplify End; End-Start;Simplified Expression Lenght CSHARP"))
            //        continue;
            //    string[] data = line.Split(';');

            //    double
            //    x1 = Convert.ToDouble(data[0]),
            //    x2 = Convert.ToDouble(data[1]),
            //    x3 = Convert.ToDouble(data[2]),
            //    x4 = Convert.ToDouble(data[3]),
            //    x5 = Convert.ToDouble(data[7]);
            //    double
            //    y1 = TimeSpan.Parse(data[6]).Ticks;

            //    datas.Add
            //        (
            //            new Data
            //            (
            //                Convert.ToInt32(data[0]),
            //                Convert.ToInt32(data[1]),
            //                Convert.ToInt32(data[2]),
            //                Convert.ToInt32(data[3]),
            //                DateTime.Parse(data[4]),
            //                DateTime.Parse(data[5]),
            //                TimeSpan.Parse(data[6]),
            //                Convert.ToInt32(data[7])
            //            )
            //         );

            //    in_x1 = in_x1.Append(x1);
            //    out_x2 = out_x2.Append(x2);
            //    highOut_x3 = highOut_x3.Append(x3);
            //    expQntd_x4 = expQntd_x4.Append(x4);
            //    simpExpQntd_x5 = simpExpQntd_x5.Append(x5);
            //    ttl_y1 = ttl_y1.Append(y1);

            //    features.Add(new double[] { x1, x2, x3, x4, x5 });
            //}

            ////double[][] features = new double[][]
            ////{
            ////    in_x1.ToArray(),
            ////    out_x2.ToArray(),
            ////    highOut_x3.ToArray(),
            ////    expQntd_x4.ToArray(),
            ////    simpExpQntd_x5.ToArray(),
            ////};

            //double[] output = ttl_y1.ToArray();

            //DateTime a, b, c, d;
            //a = DateTime.Now;
            //MachineLearning.MLDatas mlDatas = new MachineLearning.MLDatas(features.ToArray(), output, 60, 20, 20);
            //b = DateTime.Now;
            //c = DateTime.Now;
            //MachineLearning.MLDatas cudamlDatas = new MachineLearning.CUDAMLDatas(features.ToArray(), output, 60, 20, 20);
            //d = DateTime.Now;

            //Console.WriteLine
            //(
            //    $"\nElements count:{mlDatas.ElementsCount}" +
            //    $"\nFeatures count:{mlDatas.FeaturesCount}" +
            //    $"\nTrain elements count:{mlDatas.TrainElementsCount}" +
            //    $"\nTest elements count:{mlDatas.TestElementsCount}" +
            //    $"\nDevelopment elements count:{mlDatas.DevelopmentElementsCount}" +
            //    $"\nTrain proportion:{mlDatas.TrainProportion}" +
            //    $"\nTest proportion:{mlDatas.TestProportion}" +
            //    $"\nDevelopment proportion:{mlDatas.DevelopmentProportion}" +
            //    $"\nStart:{a}" +
            //    $"\nEnd:{b}" +
            //    $"\nTotal:{b.Subtract(a)}" +
            //    $"\n\n\n\n"
            //);
            //Console.WriteLine
            //(
            //    $"\nElements count:{cudamlDatas.ElementsCount}" +
            //    $"\nFeatures count:{cudamlDatas.FeaturesCount}" +
            //    $"\nTrain elements count:{cudamlDatas.TrainElementsCount}" +
            //    $"\nTest elements count:{cudamlDatas.TestElementsCount}" +
            //    $"\nDevelopment elements count:{cudamlDatas.DevelopmentElementsCount}" +
            //    $"\nTrain proportion:{cudamlDatas.TrainProportion}" +
            //    $"\nTest proportion:{cudamlDatas.TestProportion}" +
            //    $"\nDevelopment proportion:{cudamlDatas.DevelopmentProportion}" +
            //    $"\nStart:{c}" +
            //    $"\nEnd:{d}" +
            //    $"\nTotal:{d.Subtract(c)}" +
            //    $"\n\n\n\n"
            //);

            //int limit = 110;
            //Console.WriteLine("TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN");
            //for (int i = 0; i < cudamlDatas.TrainElementsCount; i++)
            //{
            //    string t1 = string.Empty, t2 = string.Empty;
            //    for (int j = 0; j < cudamlDatas.FeaturesCount; j++)
            //    {
            //        t1 += string.IsNullOrEmpty(t1) ? $"x{j}: {mlDatas.TrainFeatures[i][j]}" : $";   x{j}: {mlDatas.TrainFeatures[i][j]}";
            //        t2 += string.IsNullOrEmpty(t2) ? $"x{j}: {cudamlDatas.TrainFeatures[i][j]}" : $";   x{j}: {cudamlDatas.TrainFeatures[i][j]}";
            //    }
            //    t1 += $"     y: {mlDatas.TrainOutputs[i]}";
            //    t2 += $"     y: {cudamlDatas.TrainOutputs[i]}";
            //    Console.WriteLine(t1 + CreatEmptySpace(limit - t1.Length) + t2);

            //}
            //Console.ReadKey();
            //Console.WriteLine("TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST");
            //for (int i = 0; i < cudamlDatas.TestElementsCount; i++)
            //{
            //    string t1 = string.Empty, t2 = string.Empty;
            //    for (int j = 0; j < cudamlDatas.FeaturesCount; j++)
            //    {
            //        t1 += string.IsNullOrEmpty(t1) ? $"x{j}: {mlDatas.TestFeatures[i][j]}" : $";   x{j}: {mlDatas.TestFeatures[i][j]}";
            //        t2 += string.IsNullOrEmpty(t2) ? $"x{j}: {cudamlDatas.TestFeatures[i][j]}" : $";   x{j}: {cudamlDatas.TestFeatures[i][j]}";
            //    }
            //    t1 += $"     y: {mlDatas.TestOutputs[i]}";
            //    t2 += $"     y: {cudamlDatas.TestOutputs[i]}";
            //    Console.WriteLine(t1 + CreatEmptySpace(limit - t1.Length) + t2);
            //}
            //Console.ReadKey();
            //Console.WriteLine("DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT DEVELOPMENT");
            //for (int i = 0; i < cudamlDatas.DevelopmentElementsCount; i++)
            //{
            //    string t1 = string.Empty, t2 = string.Empty;
            //    for (int j = 0; j < cudamlDatas.FeaturesCount; j++)
            //    {
            //        t1 += string.IsNullOrEmpty(t1) ? $"x{j}: {mlDatas.DevelopmentFeatures[i][j]}" : $";   x{j}: {mlDatas.DevelopmentFeatures[i][j]}";
            //        t2 += string.IsNullOrEmpty(t2) ? $"x{j}: {cudamlDatas.DevelopmentFeatures[i][j]}" : $";   x{j}: {cudamlDatas.DevelopmentFeatures[i][j]}";
            //    }
            //    t1 += $"     y: {mlDatas.DevelopmentOutputs[i]}";
            //    t2 += $"     y: {cudamlDatas.DevelopmentOutputs[i]}";
            //    Console.WriteLine(t1 + CreatEmptySpace(limit - t1.Length) + t2);
            //}
            //Console.WriteLine("ok");
            //Console.ReadKey();
            #endregion

            #region Verificando as melhores opções para se adicionar novos elementos para um matriz, enquanto se remove a sua referência (2 normais 1 CUDA)
            //DateTime[] s = new DateTime[4], e = new DateTime[4];

            //Random rnd = new Random();

            //int a = 10200000,
            //    b = 10200000,
            //    c = a + b,
            //    d = 10;
            //byte[][]
            //    a1 = new byte[a][],
            //    a2 = new byte[b][];

            //IEnumerable<byte[]>
            //    a3;

            //Parallel.For(0, a > b ? a : b, i => { if (i < a) { a1[i] = new byte[d]; rnd.NextBytes(a1[i]); } if (i < b) { a2[i] = new byte[d]; rnd.NextBytes(a2[i]); } });

            //for (int i = 0; i < 4; i++)
            //{
            //    s[i] = DateTime.Now;
            //    switch (i)
            //    {
            //        case 0: //Utiliza o método Append <- Lento    
            //            a3 = a1;
            //            for (int j = 0; j < b; j++)
            //                a3 = a3.Append(a2[j].Clone() as byte[]);
            //            break;
            //        case 1: //Cria uma nova matriz e coloca tudo de novo  <- Mais rápido sem usar CUDA
            //            a3 = new byte[c][];
            //            for (int j = 0; j < c; j++)
            //                if (j < a)
            //                    ((byte[][])a3)[j] = a1[j].Clone() as byte[];
            //                else
            //                    ((byte[][])a3)[j] = a2[j - a].Clone() as byte[];
            //            break;
            //        case 2: //Cria uma nova matriz e coloca tudo de novo (Usando CUDA) <- Mais rápido (consome ~ 68,053851% do tempo gasto no 'Mais rápido sem usar CUDA')
            //            a3 = new byte[c][];
            //            Parallel.For(0, c, j => { if (j < a) ((byte[][])a3)[j] = a1[j].Clone() as byte[]; else ((byte[][])a3)[j] = a2[j - a].Clone() as byte[]; });
            //            break;
            //        //case 3: //Cria uma nova matriz e coloca tudo de novo (Usando CUDA) <- EVITAR AO MÁXIMO ESSA ABORDAGEM, EM TEORIA ELE REQUISITA UM CONJUNTO IMENSO DE NÚCLEOS DE PROCESSAMENTO
            //        //    a3 = new byte[c][];
            //        //    Parallel.For(0, c, j => { if (j < a) { ((byte[][])a3)[j] = new byte[a1[j].Length]; Parallel.For(0, a1[j].Length, k => ((byte[][])a3)[j][k] = a1[j][k]); } else { ((byte[][])a3)[j] = new byte[a2[j - a].Length]; Parallel.For(0, a2[j - a].Length, k => ((byte[][])a3)[j][k] = a2[j - a][k]); } });
            //        //    break;
            //    }
            //    e[i] = DateTime.Now;
            //    Console.WriteLine($"Test {i}:\nStart: {s[i]}\nEnd: {e[i]}\nTotal: {e[i].Subtract(s[i])}\n\n");
            //}
            #endregion

            #region Testando desvinculação de referência
            //byte[][] a = new byte[3][]
            //    {
            //        new byte[]{1,2,3 },
            //        new byte[]{1,2,3 },
            //        new byte[]{1,2,3 },
            //    }, b = a;
            //Console.WriteLine("[" + string.Join("]\n[", a.ToList().ConvertAll((ar) => string.Join(";", ar))) + "]" + "\n\n");
            //Console.WriteLine("[" + string.Join("]\n[", b.ToList().ConvertAll((ar) => string.Join(";", ar))) + "]" + "\n\n");

            //a = new byte[2][]
            //    {
            //        new byte[]{10,9,8},
            //        new byte[]{10,9,8},
            //    };
            //Console.WriteLine("[" + string.Join("]\n[", a.ToList().ConvertAll((ar) => string.Join(";", ar))) + "]" + "\n\n");
            //Console.WriteLine("[" + string.Join("]\n[", b.ToList().ConvertAll((ar) => string.Join(";", ar))) + "]" + "\n\n");
            #endregion
            Console.ReadKey();

        }

        static string WriteArray(double[] array)
        {
            string t = "{";
            foreach(double x in array)
                t += t=="{"? $"{x}": $";{x}";
            t += "}";
            return t;
        }
        static string WriteArray(double[][] array, string previousText = "")
        {
            string t = string.Empty;
            for (int i = 0; i < array.Length; i++)
                t += string.IsNullOrEmpty(t) ? $"{previousText}[{i}]{WriteArray(array[i])}" : $"\n{previousText}[{i}]{WriteArray(array[i])}";
            return t;
        }
        static string WriteArray(double[][][] array)
        {
            string t = string.Empty;
            for (int i = 0; i < array.Length; i++)
                t += string.IsNullOrEmpty(t) ? $"{WriteArray(array[i],$"[{i}]")}" : $"\n{WriteArray(array[i], $"[{i}]")}";
            return t;
        }
        static string WriteArray(MachineLearning.MachineLearningTools.GeneralTools.ExpoentFeatureAddress[] array)
        {
            string t = "{";
            foreach (MachineLearning.MachineLearningTools.GeneralTools.ExpoentFeatureAddress x in array)
                t += t == "{" ? $"{x}" : $";{x}";
            t += "}";
            return t;
        }
        
        static string CreatEmptySpace(int lenght)
        {
            char[] t = new char[lenght];
            Parallel.For(0, lenght, i => t[i] += ' ');
            return string.Concat(t);
        }

        /// <summary>
        /// Get the standart deviation from a list of elements 
        /// </summary>
        /// <param name="elements"></param>
        /// <returns></returns>
        public static double GetStandartDeviation(IList<double> elements)
        {
            double avrg = elements.Average();
            double count = elements.Count();
            double sigma = 0;
            for (int i = 0; i < count; i++)
                sigma += Math.Pow(elements[i] - avrg, 2);
            return count > 1 ? Math.Sqrt(sigma / count - 1) : 0;
        }

        class Data
        {
            public double Inputs { get; private set; } = 0;
            public double Output { get; private set; } = 0;
            public double ExpressionsCount { get; private set; } = 0;
            public double ExpressionTextLenght { get; private set; } = 0;
            public DateTime SimplificationStart { get; private set; } = DateTime.MinValue;
            public DateTime SimplificationEnd { get; private set; } = DateTime.MinValue;
            public TimeSpan Total { get; private set; } = TimeSpan.MinValue;
            public double SimplifiedExpressionTextLenght { get; private set; } = 0;

            public Data(int inputs, int output, int expressionsCount, int expressionTextLenght, DateTime simplificationStart, DateTime simplificationEnd, TimeSpan total, int simplifiedExpressionTextLenght)
            {
                Inputs = inputs;
                Output = output;
                ExpressionsCount = expressionsCount;
                ExpressionTextLenght = expressionTextLenght;
                SimplificationStart = simplificationStart;
                SimplificationEnd = simplificationEnd;
                Total = total;
                SimplifiedExpressionTextLenght = simplifiedExpressionTextLenght;
            }
        }

        class Datas : IList<Data>
        {
            List<Data> DatasList { get; set; } = new List<Data>();

            public Data this[int index] { get => DatasList[index]; set => DatasList[index] = value; }

            public int Count => DatasList.Count;

            public bool IsReadOnly => false;

            public void Add(Data item)
            {
                DatasList.Add(item);
            }

            public void Clear()
            {
                DatasList.Clear();
            }

            public bool Contains(Data item)
            {
                return DatasList.Contains(item);
            }

            public void CopyTo(Data[] array, int arrayIndex)
            {
                DatasList.CopyTo(array, arrayIndex);
            }

            public IEnumerator<Data> GetEnumerator()
            {
                return DatasList.GetEnumerator();
            }

            public int IndexOf(Data item)
            {
                return DatasList.IndexOf(item);
            }

            public void Insert(int index, Data item)
            {
                DatasList.Insert(index, item);
            }

            public bool Remove(Data item)
            {
                return DatasList.Remove(item);
            }

            public void RemoveAt(int index)
            {
                DatasList.RemoveAt(index);
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return DatasList.GetEnumerator();
            }

            /// <summary>
            /// Return all spended time from all datas.
            /// </summary>
            /// <returns></returns>
            public TimeSpan TimeSpended()
            {
                TimeSpan total = TimeSpan.MinValue;
                foreach (Data data in DatasList)
                    if (total == TimeSpan.MinValue)
                        total = data.Total;
                    else
                        total = total.Add(data.Total);

                return total;
            }
        }
    }
}





