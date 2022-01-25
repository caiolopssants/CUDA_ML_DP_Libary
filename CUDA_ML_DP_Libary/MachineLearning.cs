using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Hybridizer;
using System.Collections;
using System.Collections.ObjectModel;
namespace CUDA_ML_Libary
{
    /// <summary>
    /// Machine learning functionality collection
    /// </summary>
    public static class MachineLearning
    {
        /// <summary>
        /// Class to storeage data informations for machine learning implementation.
        /// <para><see cref="double"/>[Lenght: Elements Count][Lenght: Features Count] All Elements/ Train Elements/ Test Elements/ Development Elements</para>
        /// <para><see cref="double"/>[Lenght: Elements Count][Lenght: Outputs Count] All Outputs/ Train Outputs/ Test Outputs/ Development Outputs</para>
        /// </summary>
        public class MLDatas
        {
            /// <summary>
            /// Array with all features
            /// </summary>
            public double[][] Features { get; protected set; } = new double[][] { };
            /// <summary>
            /// Features lenght
            /// </summary>
            public int FeaturesCount { get { return Features.FirstOrDefault() is double[] arr ? arr.Length : 0; } }
            /// <summary>
            /// Elements count
            /// </summary>
            public int ElementsCount { get { return Features.Length; } }

            /// <summary>
            /// Array with all outputs from cost function class
            /// </summary>
            public double[][] Outputs { get; protected set; } = new double[][] { };


            /// <summary>
            /// Test array proportion value
            /// </summary>
            public double TestProportion { get; protected set; } = 0;
            /// <summary>
            /// Test array
            /// </summary>
            public double[][] TestFeatures { get; protected set; } = new double[][] { };
            /// <summary>
            /// Test output array
            /// </summary>
            public double[][] TestOutputs { get; protected set; } = new double[][] { };
            /// <summary>
            /// Test Elements count
            /// </summary>
            public int TestElementsCount { get; protected set; }

            /// <summary>
            /// Train array proportion value
            /// </summary>
            public double TrainProportion { get; protected set; } = 0;
            /// <summary>
            /// Train array
            /// </summary>
            public double[][] TrainFeatures { get; protected set; } = new double[][] { };
            /// <summary>
            /// Train outputs array
            /// </summary>
            public double[][] TrainOutputs { get; protected set; } = new double[][] { };
            /// <summary>
            /// Train Elements count
            /// </summary>
            public int TrainElementsCount { get; protected set; }

            /// <summary>
            /// Development array proportion value
            /// </summary>
            public double DevelopmentProportion { get; protected set; } = 0;
            /// <summary>
            /// Development array
            /// </summary>
            public double[][] DevelopmentFeatures { get; protected set; } = new double[][] { };
            /// <summary>
            /// Development output array
            /// </summary>
            public double[][] DevelopmentOutputs { get; protected set; } = new double[][] { };
            /// <summary>
            /// Development Elements count
            /// </summary>
            public int DevelopmentElementsCount { get; protected set; }
            
            /// <summary>
            /// Creat a Machine Learning data storeage instance
            /// </summary>
            public MLDatas()
            { }

            /// <summary>
            /// Creat a Machine Learning data storeage
            /// </summary>
            /// <param name="features">Data features</param>
            /// <param name="output">Data outputs</param>        
            /// <param name="trainProportion">Train proportion.<para>Default value: 70 %</para></param>
            /// <param name="testProportion">Test proportion.<para>Default value: 15 %</para></param>
            /// <param name="developmentProportion">Development proportion.<para>Default value: 15 %</para></param>
            /// <exception cref="ArgumentOutOfRangeException"></exception>        
            /// <exception cref="ArgumentException"></exception>
            public MLDatas(double[][] features,
                         double[] output,
                         uint trainProportion = 70,
                         uint testProportion = 15,
                         uint developmentProportion = 15)
            {
                if (features.Length != output.Length)
                    throw new ArgumentOutOfRangeException($"{nameof(features)} length {{{features.Length}}} it is different from  {nameof(output)} length {{{output.Length}}}");

                Features = new double[features.Length][];
                Outputs = new double[output.Length][];
                for (int i = 0; i < features.Length; i++)
                {
                    Features[i] = features[i].Clone() as double[];
                    Outputs[i] = new double[] { output[i] }; 
                }

                SetProportions(trainProportion, testProportion, developmentProportion);
                ShuffleSets();
            }
            /// <summary>
            /// Creat a Machine Learning data storeage
            /// </summary>
            /// <param name="features">Data features</param>
            /// <param name="output">Data outputs</param>        
            /// <param name="trainProportion">Train proportion.<para>Default value: 70 %</para></param>
            /// <param name="testProportion">Test proportion.<para>Default value: 15 %</para></param>
            /// <param name="developmentProportion">Development proportion.<para>Default value: 15 %</para></param>
            /// <exception cref="ArgumentOutOfRangeException"></exception>        
            /// <exception cref="ArgumentException"></exception>
            public MLDatas(double[][] features,
                         double[][] output,
                         uint trainProportion = 70,
                         uint testProportion = 15,
                         uint developmentProportion = 15)
            {
                if (features.Length != output.Length)
                    throw new ArgumentOutOfRangeException($"{nameof(features)} length {{{features.Length}}} it is different from  {nameof(output)} length {{{output.Length}}}");

                Features = new double[features.Length][];
                Outputs = new double[output.Length][];

                for (int i = 0; i < features.Length; i++)
                {
                    Features[i] = features[i].Clone() as double[];
                    Outputs[i] = output[i].Clone() as double[];
                }


                SetProportions(trainProportion, testProportion, developmentProportion);
                ShuffleSets();
            }

            /// <summary>
            /// Append features and output datas from other <see cref="MLDatas"/> instance
            /// </summary>
            /// <param name="datas"></param>
            public void AppendDatas(MLDatas datas)
            {
                double[][]
                    oldFea = Features,
                    oldOut = Outputs;
                int
                    aF = oldFea.Length,
                    bF = datas.Features.Length,
                    cF = aF + bF,

                    aO = oldOut.Length, 
                    bO = datas.Outputs.Length,
                    cO = aO+bO;
                
                Features = new double[cF][];
                Outputs = new double[cO][];
                for (int i = 0, j = 0; i < cO || j < cF; i++, j++)
                {
                    if (i < aO)
                        Outputs[i] = oldOut[i].Clone() as double[];
                    else if (i < cO)
                        Outputs[i] = datas.Outputs[i - aO].Clone() as double[];

                    if (j < aF)
                        Features[j] = oldFea[j].Clone() as double[];
                    else if (j < cF)
                        Features[j] = datas.Features[j - aF].Clone() as double[];
                }
                SetProportions(Convert.ToUInt32(TrainProportion), Convert.ToUInt32(TestProportion), Convert.ToUInt32(DevelopmentProportion));
                ShuffleSets();                
            }
            /// <summary>
            /// Append features and outputs from a two arrays
            /// </summary>
            /// <param name="features">Features array</param>
            /// <param name="outputs">Outputs array</param>
            public void AppendDatas(double[][] features, double[] outputs)
            {
                double[][]
                    oldFea = Features,
                    oldOut = Outputs;
                int
                    aF = oldFea.Length,
                    bF = features.Length,
                    cF = aF + bF,

                    aO = oldOut.Length,
                    bO = outputs.Length,
                    cO = aO + bO;

                Features = new double[cF][];
                Outputs = new double[cO][];
                for (int i = 0, j = 0; i < cO || j < cF; i++, j++)
                {
                    if (i < aO)
                        Outputs[i] = oldOut[i].Clone() as double[];
                    else if (i < cO)
                        Outputs[i] = new double[] { outputs[i - aO] };

                    if (j < aF)
                        Features[j] = oldFea[j].Clone() as double[];
                    else if (j < cF)
                        Features[j] = features[j - aF].Clone() as double[];
                }
                SetProportions(Convert.ToUInt32(TrainProportion), Convert.ToUInt32(TestProportion), Convert.ToUInt32(DevelopmentProportion));
                ShuffleSets();
            }
            /// <summary>
            /// Append features and outputs from a two arrays
            /// </summary>
            /// <param name="features">Features array</param>
            /// <param name="outputs">Outputs array</param>
            public void AppendDatas(double[][] features, double[][] outputs)
            {
                double[][]
                    oldFea = Features,
                    oldOut = Outputs;
                int
                    aF = oldFea.Length,
                    bF = features.Length,
                    cF = aF + bF,

                    aO = oldOut.Length,
                    bO = outputs.Length,
                    cO = aO + bO;

                Features = new double[cF][];
                Outputs = new double[cO][];
                for (int i = 0, j = 0; i < cO || j < cF; i++, j++)
                {
                    if (i < aO)
                        Outputs[i] = oldOut[i].Clone() as double[];
                    else if (i < cO)
                        Outputs[i] = outputs[i - aO].Clone() as double[];

                    if (j < aF)
                        Features[j] = oldFea[j].Clone() as double[];
                    else if (j < cF)
                        Features[j] = features[j - aF].Clone() as double[];
                }
                SetProportions(Convert.ToUInt32(TrainProportion), Convert.ToUInt32(TestProportion), Convert.ToUInt32(DevelopmentProportion));
                ShuffleSets();
            }

            /// <summary>
            /// Set new proportions and shuffle the new sets
            /// </summary>
            /// <param name="trainProportion">Train proportion.<para>Default value: 70 %</para></param>
            /// <param name="testProportion">Test proportion.<para>Default value: 15 %</para></param>
            /// <param name="developmentProportion">Development proportion.<para>Default value: 15 %</para></param>
            public void SetProportions(uint trainProportion = 70, uint testProportion = 15, uint developmentProportion = 15)
            {
                if (trainProportion + testProportion + developmentProportion != 100)
                    throw new ArgumentOutOfRangeException($"The sum of {nameof(trainProportion)} (current value: {trainProportion}) , {nameof(testProportion)} (current value: {testProportion}) and {nameof(developmentProportion)} (current value: {developmentProportion}) need to be equal 100%");


                int trainIndex = 0;
                int testIndex = 0;
                int developmentIndex = 0;

                List<int> indexes = new List<int>();
                for (int i = 0; i < ElementsCount; i++)
                    indexes.Add(i);
                Random rnd = new Random(DateTime.Now.Millisecond);
                for (int i = 0; i < ElementsCount; i++)
                {
                    if (i == 0)
                    {
                        double trainElementsLenghtDec = Math.Round(ElementsCount * Convert.ToDouble(trainProportion) / 100, 0);
                        double testElementsLenghtDec = Math.Round(ElementsCount * Convert.ToDouble(testProportion) / 100, 0);
                        double developmentElementsLenghtDec = Math.Round(ElementsCount * Convert.ToDouble(developmentProportion) / 100, 0);

                        trainElementsLenghtDec += ElementsCount - (trainElementsLenghtDec + testElementsLenghtDec + developmentElementsLenghtDec);

                        TrainElementsCount = Convert.ToInt32(trainElementsLenghtDec);
                        TestElementsCount = Convert.ToInt32(testElementsLenghtDec);
                        DevelopmentElementsCount = Convert.ToInt32(developmentElementsLenghtDec);

                        TrainFeatures = new double[TrainElementsCount][];
                        TrainOutputs = new double[TrainElementsCount][];

                        TestFeatures = new double[TestElementsCount][];
                        TestOutputs = new double[TestElementsCount][];

                        DevelopmentFeatures = new double[DevelopmentElementsCount][];
                        DevelopmentOutputs = new double[DevelopmentElementsCount][];
                    }
                    else if (FeaturesCount != Features[i].Length)
                        throw new ArgumentException($"Different feature lenght:\nLenghts\nIndex 0: {FeaturesCount}\nIndex {i}: {Features[i].Length}");


                    double sen = Math.Sin(DateTime.Now.Ticks * i / Math.E * Math.PI / 180);
                    int i_rnd = rnd.Next(0, indexes.Count);
                    if ((developmentIndex == DevelopmentElementsCount && testIndex == TestElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= 1 && sen >= 0.33333 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex == TestElementsCount) && sen < 0 ||
                                (developmentIndex == DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen < 0)
                    {
                        //Add train feature

                        TrainFeatures[trainIndex] = Features[indexes[i_rnd]];
                        TrainOutputs[trainIndex] = Outputs[indexes[i_rnd]];
                        trainIndex++;
                    }
                    else
                    {
                        if ((developmentIndex == DevelopmentElementsCount && trainIndex == TrainElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= 0.33334 && sen >= -0.33334 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex == TrainElementsCount && testIndex != TestElementsCount) && sen < 0 ||
                                (developmentIndex == DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen > 0)
                        {
                            //Add test feature
                            TestFeatures[testIndex] = Features[indexes[i_rnd]];
                            TestOutputs[testIndex] = Outputs[indexes[i_rnd]];
                            testIndex++;
                        }
                        else
                        {
                            if ((trainIndex == TrainElementsCount && testIndex == TestElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= -0.33335 && sen >= -1 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex == TrainElementsCount && testIndex != TestElementsCount) && sen > 0 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex == TestElementsCount) && sen > 0)
                            {
                                //Add development feature
                                DevelopmentFeatures[developmentIndex] = Features[indexes[i_rnd]];
                                DevelopmentOutputs[developmentIndex] = Outputs[indexes[i_rnd]];
                                developmentIndex++;
                            }
                        }
                    }
                    indexes.RemoveAt(i_rnd);
                }

                TrainProportion = trainProportion;
                TestProportion = testProportion;
                DevelopmentProportion = developmentProportion;


            }
            /// <summary>
            /// Will creat a new train, test and development sets
            /// </summary>
            public void ShuffleSets()
            {
                int trainIndex = 0;
                int testIndex = 0;
                int developmentIndex = 0;

                List<int> indexes = new List<int>();
                for (int i = 0; i < ElementsCount; i++)
                    indexes.Add(i);
                Random rnd = new Random(DateTime.Now.Millisecond);
                for (int i = 0; i < ElementsCount; i++)
                {
                    double sen = Math.Sin(DateTime.Now.Ticks * i / Math.E * Math.PI / 180);
                    int i_rnd = rnd.Next(0, indexes.Count);
                    if ((developmentIndex == DevelopmentElementsCount && testIndex == TestElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= 1 && sen >= 0.33333 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex == TestElementsCount) && sen < 0 ||
                                (developmentIndex == DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen < 0)
                    {
                        //Add train feature
                        TrainFeatures[trainIndex] = Features[indexes[i_rnd]];
                        TrainOutputs[trainIndex] = Outputs[indexes[i_rnd]];
                        trainIndex++;
                    }
                    else
                    {
                        if ((developmentIndex == DevelopmentElementsCount && trainIndex == TrainElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= 0.33334 && sen >= -0.33334 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex == TrainElementsCount && testIndex != TestElementsCount) && sen < 0 ||
                                (developmentIndex == DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen > 0)
                        {
                            //Add test feature
                            TestFeatures[testIndex] = Features[indexes[i_rnd]];
                            TestOutputs[testIndex] = Outputs[indexes[i_rnd]];
                            testIndex++;
                        }
                        else
                        {
                            if ((trainIndex == TrainElementsCount && testIndex == TestElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= -0.33335 && sen >= -1 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex == TrainElementsCount && testIndex != TestElementsCount) && sen > 0 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex == TestElementsCount) && sen > 0)
                            {
                                //Add development feature
                                DevelopmentFeatures[developmentIndex] = Features[indexes[i_rnd]];
                                DevelopmentOutputs[developmentIndex] = Outputs[indexes[i_rnd]];
                                developmentIndex++;
                            }
                        }
                    }
                    indexes.RemoveAt(i_rnd);
                }
            }

            
        }
        /// <summary>
        /// Class to storeage data informations for machine learning implementation (CUDA Hybridizer process implementation).
        /// <para><see cref="double"/>[Lenght: Elements Count][Lenght: Features Count] All Elements/ Train Elements/ Test Elements/ Development Elements</para>
        /// <para><see cref="double"/>[Lenght: Elements Count][Lenght: Outputs Count] All Outputs/ Train Outputs/ Test Outputs/ Development Outputs</para>
        /// </summary>
        public class CUDAMLDatas : MLDatas
        {
            /// <summary>
            /// Creat a Machine Learning data storeage using CUDA Hybridizer
            /// </summary>
            /// <param name="features">Data features</param>
            /// <param name="output">Data outputs</param>        
            /// <param name="trainProportion">Train proportion.<para>Default value: 70 %</para></param>
            /// <param name="testProportion">Test proportion.<para>Default value: 15 %</para></param>
            /// <param name="developmentProportion">Development proportion.<para>Default value: 15 %</para></param>
            /// <exception cref="ArgumentOutOfRangeException"></exception>        
            /// <exception cref="ArgumentException"></exception>
            public CUDAMLDatas(double[][] features,
                         double[] output,
                         uint trainProportion = 70,
                         uint testProportion = 15,
                         uint developmentProportion = 15)
            {
                if (features.Length != output.Length)
                    throw new ArgumentOutOfRangeException($"{nameof(features)} length {{{features.Length}}} it is different from  {nameof(output)} length {{{output.Length}}}");

                Features = new double[features.Length][];
                Outputs = new double[output.Length][];

                Parallel.For(0, features.Length, i => { Features[i] = features[i].Clone() as double[]; Outputs[i] = new double[] { output[i] }; });                

                CUDASetProportions(trainProportion, testProportion, developmentProportion);

                CUDAShuffleSets();
            }

            /// <summary>
            /// Creat a Machine Learning data storeage using CUDA Hybridizer
            /// </summary>
            /// <param name="features">Data features</param>
            /// <param name="output">Data outputs</param>        
            /// <param name="trainProportion">Train proportion.<para>Default value: 70 %</para></param>
            /// <param name="testProportion">Test proportion.<para>Default value: 15 %</para></param>
            /// <param name="developmentProportion">Development proportion.<para>Default value: 15 %</para></param>
            /// <exception cref="ArgumentOutOfRangeException"></exception>        
            /// <exception cref="ArgumentException"></exception>
            public CUDAMLDatas(double[][] features,
                         double[][] output,
                         uint trainProportion = 70,
                         uint testProportion = 15,
                         uint developmentProportion = 15)
            {
                if (features.Length != output.Length)
                    throw new ArgumentOutOfRangeException($"{nameof(features)} length {{{features.Length}}} it is different from  {nameof(output)} length {{{output.Length}}}");

                Features = new double[features.Length][];
                Outputs = new double[output.Length][];

                Parallel.For(0, features.Length, i => { Features[i] = features[i].Clone() as double[]; Outputs[i] = output[i].Clone() as double[]; });

                CUDASetProportions(trainProportion, testProportion, developmentProportion);

                CUDAShuffleSets();
            }

            /// <summary>
            /// Append features and output datas from other <see cref="MLDatas"/> instance using CUDA Hybridizer
            /// </summary>
            /// <param name="datas"></param>
            public void CUDAAppendDatas(MLDatas datas)
            {
                double[][]
                    oldFea = Features,
                    oldOut = Outputs;
                int
                    aF = oldFea.Length,
                    bF = datas.Features.Length,
                    cF = aF + bF,

                    aO = oldOut.Length,
                    bO = datas.Outputs.Length,
                    cO = aO + bO;

                Features = new double[cF][];
                Outputs = new double[cO][];
                Parallel.For(0, cO /*cO ou cF possuem o mesmo valor*/, i => { if (i < aO) Outputs[i] = oldOut[i].Clone() as double[]; else if (i < cO) Outputs[i] = datas.Outputs[i - aO].Clone() as double[]; if (i < aF) Features[i] = oldFea[i].Clone() as double[]; else if (i < cF) Features[i] = datas.Features[i - aF].Clone() as double[]; });

                CUDASetProportions(Convert.ToUInt32(TrainProportion), Convert.ToUInt32(TestProportion), Convert.ToUInt32(DevelopmentProportion));
                CUDAShuffleSets();
            }
            /// <summary>
            /// Append features and outputs from a two arrays using CUDA Hybridizer
            /// </summary>
            /// <param name="features">Features array</param>
            /// <param name="outputs">Outputs array</param>
            public void CUDAAppendDatas(double[][] features, double[] outputs)
            {
                double[][]
                    oldFea = Features,
                    oldOut = Outputs;
                int
                    aF = oldFea.Length,
                    bF = features.Length,
                    cF = aF + bF,

                    aO = oldOut.Length,
                    bO = outputs.Length,
                    cO = aO + bO;

                Features = new double[cF][];
                Outputs = new double[cO][];
                Parallel.For(0, cO /*cO ou cF possuem o mesmo valor*/, i => { if (i < aO) Outputs[i] = oldOut[i].Clone() as double[]; else if (i < cO) Outputs[i] = new double[] { outputs[i - aO] }; if (i < aF) Features[i] = oldFea[i].Clone() as double[]; else if (i < cF) Features[i] = features[i - aF].Clone() as double[]; });
                CUDASetProportions(Convert.ToUInt32(TrainProportion), Convert.ToUInt32(TestProportion), Convert.ToUInt32(DevelopmentProportion));
                CUDAShuffleSets();
            }
            /// <summary>
            /// Append features and outputs from a two arrays using CUDA Hybridizer
            /// </summary>
            /// <param name="features">Features array</param>
            /// <param name="outputs">Outputs array</param>ame="outputs">Outputs array</param>
            public void CUDAAppendDatas(double[][] features, double[][] outputs)
            {
                double[][]
                    oldFea = Features,
                    oldOut = Outputs;
                int
                    aF = oldFea.Length,
                    bF = features.Length,
                    cF = aF + bF,

                    aO = oldOut.Length,
                    bO = outputs.Length,
                    cO = aO + bO;

                Features = new double[cF][];
                Outputs = new double[cO][];
                Parallel.For(0, cO /*cO ou cF possuem o mesmo valor*/, i => { if (i < aO) Outputs[i] = oldOut[i].Clone() as double[]; else if (i < cO) Outputs[i] = outputs[i - aO].Clone() as double[]; if (i < aF) Features[i] = oldFea[i].Clone() as double[]; else if (i < cF) Features[i] = features[i - aF].Clone() as double[]; });

                CUDASetProportions(Convert.ToUInt32(TrainProportion), Convert.ToUInt32(TestProportion), Convert.ToUInt32(DevelopmentProportion));
                CUDAShuffleSets();
            }

            /// <summary>
            /// Set new proportions and shuffle the new sets using CUDA Hybridizer
            /// </summary>
            /// <param name="trainProportion">Train proportion.<para>Default value: 70 %</para></param>
            /// <param name="testProportion">Test proportion.<para>Default value: 15 %</para></param>
            /// <param name="developmentProportion">Development proportion.<para>Default value: 15 %</para></param>
            public void CUDASetProportions(uint trainProportion = 70, uint testProportion = 15, uint developmentProportion = 15)
            {
                if (trainProportion + testProportion + developmentProportion != 100)
                    throw new ArgumentOutOfRangeException($"The sum of {nameof(trainProportion)} (current value: {trainProportion}) , {nameof(testProportion)} (current value: {testProportion}) and {nameof(developmentProportion)} (current value: {developmentProportion}) need to be equal 100%");


                int trainIndex = 0;
                int testIndex = 0;
                int developmentIndex = 0;

                List<int> indexes = new List<int>();
                for (int i = 0; i < ElementsCount; i++)
                    indexes.Add(i);
                Random rnd = new Random(DateTime.Now.Millisecond);

                for (int i = 0; i < ElementsCount; i++)
                {
                    if (i == 0)
                    {
                        double trainElementsLenghtDec = Math.Round(ElementsCount * Convert.ToDouble(trainProportion) / 100, 0);
                        double testElementsLenghtDec = Math.Round(ElementsCount * Convert.ToDouble(testProportion) / 100, 0);
                        double developmentElementsLenghtDec = Math.Round(ElementsCount * Convert.ToDouble(developmentProportion) / 100, 0);

                        trainElementsLenghtDec += ElementsCount - (trainElementsLenghtDec + testElementsLenghtDec + developmentElementsLenghtDec);

                        TrainElementsCount = Convert.ToInt32(trainElementsLenghtDec);
                        TestElementsCount = Convert.ToInt32(testElementsLenghtDec);
                        DevelopmentElementsCount = Convert.ToInt32(developmentElementsLenghtDec);

                        TrainFeatures = new double[TrainElementsCount][];
                        TrainOutputs = new double[TrainElementsCount][];

                        TestFeatures = new double[TestElementsCount][];
                        TestOutputs = new double[TestElementsCount][];

                        DevelopmentFeatures = new double[DevelopmentElementsCount][];
                        DevelopmentOutputs = new double[DevelopmentElementsCount][];
                    }
                    else if (FeaturesCount != Features[i].Length)
                        throw new ArgumentException($"Different feature lenght:\nLenghts\nIndex 0: {FeaturesCount}\nIndex {i}: {Features[i].Length}");


                    double sen = Math.Sin(DateTime.Now.Ticks * i / Math.E * Math.PI / 180);
                    int i_rnd = rnd.Next(0, indexes.Count);
                    if ((developmentIndex == DevelopmentElementsCount && testIndex == TestElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= 1 && sen >= 0.33333 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex == TestElementsCount) && sen < 0 ||
                                (developmentIndex == DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen < 0)
                    {
                        //Add train feature

                        TrainFeatures[trainIndex] = Features[indexes[i_rnd]];
                        TrainOutputs[trainIndex] = Outputs[indexes[i_rnd]];
                        trainIndex++;
                    }
                    else
                    {
                        if ((developmentIndex == DevelopmentElementsCount && trainIndex == TrainElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= 0.33334 && sen >= -0.33334 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex == TrainElementsCount && testIndex != TestElementsCount) && sen < 0 ||
                                (developmentIndex == DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen > 0)
                        {
                            //Add test feature
                            TestFeatures[testIndex] = Features[indexes[i_rnd]];
                            TestOutputs[testIndex] = Outputs[indexes[i_rnd]];
                            testIndex++;
                        }
                        else
                        {
                            if ((trainIndex == TrainElementsCount && testIndex == TestElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= -0.33335 && sen >= -1 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex == TrainElementsCount && testIndex != TestElementsCount) && sen > 0 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex == TestElementsCount) && sen > 0)
                            {
                                //Add development feature
                                DevelopmentFeatures[developmentIndex] = Features[indexes[i_rnd]];
                                DevelopmentOutputs[developmentIndex] = Outputs[indexes[i_rnd]];
                                developmentIndex++;
                            }
                        }
                    }
                    indexes.RemoveAt(i_rnd);
                }

                TrainProportion = trainProportion;
                TestProportion = testProportion;
                DevelopmentProportion = developmentProportion;


            }
            /// <summary>
            /// Will creat a new train, test and development sets using CUDA Hybridizer
            /// </summary>
            public void CUDAShuffleSets()
            {
                int trainIndex = 0;
                int testIndex = 0;
                int developmentIndex = 0;

                List<int> indexes = new int[ElementsCount].ToList();
                Parallel.For(0, ElementsCount, i => indexes[i] = i);
                Random rnd = new Random(DateTime.Now.Millisecond);
                for (int i = 0; i < ElementsCount; i++)
                {
                    double sen = Math.Sin(DateTime.Now.Ticks * i / Math.E * Math.PI / 180);
                    int i_rnd = rnd.Next(0, indexes.Count);
                    if ((developmentIndex == DevelopmentElementsCount && testIndex == TestElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= 1 && sen >= 0.33333 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex == TestElementsCount) && sen < 0 ||
                                (developmentIndex == DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen < 0)
                    {
                        //Add train feature
                        TrainFeatures[trainIndex] = Features[indexes[i_rnd]];
                        TrainOutputs[trainIndex] = Outputs[indexes[i_rnd]];
                        trainIndex++;
                    }
                    else
                    {
                        if ((developmentIndex == DevelopmentElementsCount && trainIndex == TrainElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= 0.33334 && sen >= -0.33334 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex == TrainElementsCount && testIndex != TestElementsCount) && sen < 0 ||
                                (developmentIndex == DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen > 0)
                        {
                            //Add test feature
                            TestFeatures[testIndex] = Features[indexes[i_rnd]];
                            TestOutputs[testIndex] = Outputs[indexes[i_rnd]];
                            testIndex++;
                        }
                        else
                        {
                            if ((trainIndex == TrainElementsCount && testIndex == TestElementsCount) ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex != TestElementsCount) && sen <= -0.33335 && sen >= -1 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex == TrainElementsCount && testIndex != TestElementsCount) && sen > 0 ||
                                (developmentIndex != DevelopmentElementsCount && trainIndex != TrainElementsCount && testIndex == TestElementsCount) && sen > 0)
                            {
                                //Add development feature
                                DevelopmentFeatures[developmentIndex] = Features[indexes[i_rnd]];
                                DevelopmentOutputs[developmentIndex] = Outputs[indexes[i_rnd]];
                                developmentIndex++;
                            }
                        }
                    }
                    indexes.RemoveAt(i_rnd);
                }
            }

            
        }


        /// <summary>
        /// Supervisioned learning collection
        /// </summary>
        public static class SupervisedLearning
        {
            /// <summary>
            /// Linear regression class
            /// </summary>
            public class LinearRegression
            {
                /// <summary>
                /// Linear regression datas
                /// <para>*Warning: Linear Regression class will pleny work when <see cref="MLDatas.JaggedArrayOutputs"/> propriety has False value, because will be used only the 0 index from jagged array </para>
                /// </summary>
                public MLDatas Datas { get; private set; }
                /// <summary>
                /// Cost function weight array
                /// </summary>
                public double[] Theta { get; private set; }
                /// <summary>
                /// Learning rate constant
                /// <para>Small value -> More slow to reach global minimum (if too small, it's possible can't reach the global minimun)</para>
                /// <para>High value -> More fast to reach global minimum (if too high, it's possible surpass the global minimun)</para>
                /// </summary>
                public double Alpha { get; set; }
                /// <summary>
                /// Constant for gradient descent regularization
                /// </summary>
                public double Lambda { get; set; }
                /// <summary>
                /// Creat a new linear regression instance
                /// </summary>
                public LinearRegression() { }
                /// <summary>
                /// Creat a new linear regression instance
                /// </summary>
                /// <param name="datas">Datas for train, test and development
                /// <para>*Warning: Linear Regression class will pleny work when <see cref="MLDatas.JaggedArrayOutputs"/> propriety has False value, because will be used only the 0 index from jagged array </para></param>
                /// <param name="alpha"></param>
                /// <param name="lambda"></param>
                /// <param name="cudaFill">Fill <see cref="Theta"/> array using <see cref="MachineLearningTools.GeneralTools.CUDAFillArray(double[], bool)"/></param>
                public LinearRegression(MLDatas datas, double alpha, double lambda, bool cudaFill)
                {
                    Datas = datas;
                    Alpha = alpha;
                    Lambda = lambda;
                    Theta = new double[datas.FeaturesCount+1];
                    if(cudaFill)
                        Theta = MachineLearningTools.GeneralTools.CUDAFillArray(Theta, false);
                    else
                        Theta = MachineLearningTools.GeneralTools.FillArray(Theta, false);                    
                }

                /// <summary>
                /// Gradient descent process using all training elements from <see cref="MLDatas.TrainFeatures"/>. 
                /// </summary>
                /// <param name="interactions">Interacions quantity to minimize <see cref="Theta"/> propriety</param>
                /// <param name="scaleAdjust">Adjust each element from a <see cref="double"/>[] array dividing by max value from this array usign the following expression:<para>array[i] = array[i]/ max(array)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* max(array) -> Max value founded on "array" array</para></para></param>
                /// <param name="normMethod">Adjust each element from a <see cref="double"/>[] by:<para><see cref="Normalization.MaxMin_Normalization"/> -> Subtracting the value by the mean and dividing by the result from subtraction between maximum value and minimum value usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ (maximum - minimum)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para> maximum/ minimum -> Maximum/ Minimum value between each features from "array"</para></para> <para><see cref="Normalization.Standardisation"/> -> Subtracting the value by the mean and dividing by the standard deviation usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ √({m|i=0} Σ(array[i] - average(array))²/(m - 1))</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para>* m -> "array" elements count</para></para></param>
                /// <param name="regulatization">Apply lambda regularization constant on gradient descent process</param>
                /// <param name="resetThetas">true - Set the 0 value for each <see cref="Theta"/> array element<para>false - use the currently values for <see cref="Theta"/></para></param>
                public void DoBatchGradientDescent(uint interactions, bool scaleAdjust = true, Normalization normMethod = Normalization.MaxMin_Normalization, bool regulatization = true, bool resetThetas = true)
                {
                    double trainElementsCount = Datas.TrainElementsCount;
                    double[] outputs = MachineLearningTools.GeneralTools.GetSpecificElements(0, Datas.TrainOutputs);

                    double
                        alpha = Alpha,
                        lambda = Lambda;

                    double[][] features;
                    if (scaleAdjust && normMethod == Normalization.None)
                        features = MachineLearningTools.Regularization.ScaleAdjust(Datas.TrainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.Standardisation)
                        features = MachineLearningTools.Regularization.Standardisation(Datas.TrainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                        features = MachineLearningTools.Regularization.MaxMinNormalisation(Datas.TrainFeatures);
                    else if (scaleAdjust && normMethod == Normalization.Standardisation)
                    { features = MachineLearningTools.Regularization.ScaleAdjust(Datas.TrainFeatures); MachineLearningTools.Regularization.Standardisation(ref features); }
                    else if (scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                    { features = MachineLearningTools.Regularization.ScaleAdjust(Datas.TrainFeatures); MachineLearningTools.Regularization.MaxMinNormalisation(ref features); }
                    else
                        features = Datas.TrainFeatures;
                    
                    double[] theta = Theta.Clone() as double[]; 
                    if (resetThetas && Theta != null)
                        MachineLearningTools.GeneralTools.FillArray(ref theta, false);

                    Func<double[][], double[], double[], int, double> sum = (ftrs, thts, outpts, ftrIndx) => // Obtendo valor da expresão  Σ(hθ(x[m]) - y[m]).x[m][ftrIndx] --> x[m][0] = 1 
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? predict - outpts[m] : (predict - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    };

                    if (regulatization)//Do gradient descent with lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            for (int j = 0; j < newTheta.Length; j++)
                                newTheta[j] = newTheta[j]*(1-(alpha*lambda/trainElementsCount)) - (alpha / trainElementsCount) * sum(features, theta, outputs, j);
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    else//Do gradient descent without lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            for (int j = 0; j < newTheta.Length; j++)
                                newTheta[j] -= alpha / trainElementsCount * sum(features, theta, outputs, j);
                            theta = newTheta.Clone() as double[];                            
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    Theta = new double[] { }.Concat(theta).ToArray();                    
                }
                /// <summary>
                /// Gradient descent process using all training elements from <see cref="MLDatas.TrainFeatures"/>. 
                /// </summary>
                /// <param name="interactions">Interacions quantity to minimize <see cref="Theta"/> propriety</param>
                /// <param name="scaleAdjust">Adjust each element from a <see cref="double"/>[] array dividing by max value from this array usign the following expression:<para>array[i] = array[i]/ max(array)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* max(array) -> Max value founded on "array" array</para></para></param>
                /// <param name="normMethod">Adjust each element from a <see cref="double"/>[] by:<para><see cref="Normalization.MaxMin_Normalization"/> -> Subtracting the value by the mean and dividing by the result from subtraction between maximum value and minimum value usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ (maximum - minimum)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para> maximum/ minimum -> Maximum/ Minimum value between each features from "array"</para></para> <para><see cref="Normalization.Standardisation"/> -> Subtracting the value by the mean and dividing by the standard deviation usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ √({m|i=0} Σ(array[i] - average(array))²/(m - 1))</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para>* m -> "array" elements count</para></para></param>
                /// <param name="regulatization">Apply lambda regularization constant on gradient descent process</param>
                /// <param name="resetThetas">true - Set the 0 value for each <see cref="Theta"/> array element<para>false - use the currently values for <see cref="Theta"/></para></param>
                public void CUDADoBatchGradientDescent(uint interactions, bool scaleAdjust = true, Normalization normMethod = Normalization.MaxMin_Normalization, bool regulatization = true, bool resetThetas = true)
                {
                    double trainElementsCount = Datas.TrainElementsCount;
                    double[] outputs = MachineLearningTools.GeneralTools.CUDAGetSpecificElements(0, Datas.TrainOutputs); ;

                    double
                        alpha = Alpha,
                        lambda = Lambda;

                    double[][] features;
                    

                    if (scaleAdjust && normMethod == Normalization.None)
                        features = MachineLearningTools.Regularization.CUDAScaleAdjust(Datas.TrainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.Standardisation)
                        features = MachineLearningTools.Regularization.CUDAStandardisation(Datas.TrainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                        features = MachineLearningTools.Regularization.CUDAMaxMinNormalisation(Datas.TrainFeatures);
                    else if (scaleAdjust && normMethod == Normalization.Standardisation)
                    { features = MachineLearningTools.Regularization.CUDAScaleAdjust(Datas.TrainFeatures); MachineLearningTools.Regularization.CUDAStandardisation(ref features); }
                    else if (scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                    { features = MachineLearningTools.Regularization.CUDAScaleAdjust(Datas.TrainFeatures); MachineLearningTools.Regularization.CUDAMaxMinNormalisation(ref features); }
                    else
                        features = Datas.TrainFeatures;

                    double[] theta = Theta.Clone() as double[];
                    if (resetThetas && Theta != null)
                        MachineLearningTools.GeneralTools.CUDAFillArray(ref theta, false);


                    Func<double[][], double[], double[], int, double> sum = (ftrs, thts, outpts, ftrIndx) => // Obtendo valor da expresão  Σ(hθ(x[m]) - y[m]).x[m][ftrIndx] --> x[m][0] = 1 
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? predict - outpts[m] : (predict - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    };


                    if (regulatization)//Do gradient descent with lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            Parallel.For(0, newTheta.Length, j => { newTheta[j] = newTheta[j] * (1 - (alpha * lambda / trainElementsCount)) - (alpha / trainElementsCount) * sum(features, theta, outputs, j); });
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    else//Do gradient descent without lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            Parallel.For(0, newTheta.Length, j => { newTheta[j] -= alpha / trainElementsCount * sum(features, theta, outputs, j); });
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    Theta = new double[] { }.Concat(theta).ToArray();
                }


                /// <summary>
                /// Normalizations methods
                /// </summary>
                public enum Normalization
                {
                    /// <summary>
                    /// MaxMinNormalisation or CUDAMaxMinNormalisation methods
                    /// </summary>
                    MaxMin_Normalization,
                    /// <summary>
                    /// Standardisation or CUDAStandardisation methods
                    /// </summary>
                    Standardisation,
                    /// <summary>
                    /// 
                    /// </summary>
                    None
                }
            }
            /// <summary>
            /// Logistic regression class
            /// </summary>
            public class LogisticRegression
            {
                /// <summary>
                /// Logist regression datas
                /// <para>*Warning: Logistic Regression class will pleny work when <see cref="MLDatas.JaggedArrayOutputs"/> propriety has False value, because will be used only the 0 index from jagged array </para>
                /// </summary>
                public MLDatas Datas { get; private set; }
                /// <summary>
                /// Cost function weight array
                /// </summary>
                public double[] Theta { get; private set; }
                /// <summary>
                /// Learning rate constant
                /// <para>Small value -> More slow to reach global minimum (if too small, it's possible can't reach the global minimun)</para>
                /// <para>High value -> More fast to reach global minimum (if too high, it's possible surpass the global minimun)</para>
                /// </summary>
                public double Alpha { get; set; }
                /// <summary>
                /// Constant for gradient descent regularization
                /// </summary>
                public double Lambda { get; set; }

                /// <summary>
                /// Creat a new logistic regression instance
                /// </summary>
                public LogisticRegression() { }
                /// <summary>
                /// Creat a new logistic regression instance
                /// </summary>
                /// <param name="datas">Datas for train, test and development
                /// <para>*Warning¹: The values from <see cref="MLDatas.Outputs"/> propriety need be seted with binary values (0 or 1)</para>
                /// <para>*Warning²: Logistic Regression class will pleny work when <see cref="MLDatas.JaggedArrayOutputs"/> propriety has False value, because will be used only the 0 index from jagged array </para></param>
                /// <param name="alpha"></param>
                /// <param name="lambda"></param>
                /// <param name="cudaFill">Fill <see cref="Theta"/> array using <see cref="MachineLearningTools.GeneralTools.CUDAFillArray(double[], bool)"/></param>
                public LogisticRegression(MLDatas datas, double alpha, double lambda, bool cudaFill)
                {
                    Datas = datas;
                    Alpha = alpha;
                    Lambda = lambda;
                    Theta = new double[datas.FeaturesCount + 1];
                    if (cudaFill)
                        Theta = MachineLearningTools.GeneralTools.CUDAFillArray(Theta, false);
                    else
                        Theta = MachineLearningTools.GeneralTools.FillArray(Theta, false);
                }

                /// <summary>
                /// Gradient descent process using all training elements from <see cref="MLDatas.TrainFeatures"/>. 
                /// </summary>
                /// <param name="interactions">Interacions quantity to minimize <see cref="Theta"/> propriety</param>
                /// <param name="scaleAdjust">Adjust each element from a <see cref="double"/>[] array dividing by max value from this array usign the following expression:<para>array[i] = array[i]/ max(array)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* max(array) -> Max value founded on "array" array</para></para></param>
                /// <param name="normMethod">Adjust each element from a <see cref="double"/>[] by:<para><see cref="Normalization.MaxMin_Normalization"/> -> Subtracting the value by the mean and dividing by the result from subtraction between maximum value and minimum value usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ (maximum - minimum)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para> maximum/ minimum -> Maximum/ Minimum value between each features from "array"</para></para> <para><see cref="Normalization.Standardisation"/> -> Subtracting the value by the mean and dividing by the standard deviation usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ √({m|i=0} Σ(array[i] - average(array))²/(m - 1))</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para>* m -> "array" elements count</para></para></param>
                /// <param name="regulatization">Apply lambda regularization constant on gradient descent process</param>
                /// <param name="resetThetas">true - Set the 0 value for each <see cref="Theta"/> array element<para>false - use the currently values for <see cref="Theta"/></para></param>
                public void DoBatchGradientDescent(uint interactions, bool scaleAdjust = true, Normalization normMethod = Normalization.MaxMin_Normalization, bool regulatization = true, bool resetThetas = true)
                {
                    double trainElementsCount = Datas.TrainElementsCount;
                    double[] outputs = MachineLearningTools.GeneralTools.GetSpecificElements(0, Datas.TrainOutputs);

                    double
                        alpha = Alpha,
                        lambda = Lambda;

                    double[][] features;
                    if (scaleAdjust && normMethod == Normalization.None)
                        features = MachineLearningTools.Regularization.ScaleAdjust(Datas.TrainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.Standardisation)
                        features = MachineLearningTools.Regularization.Standardisation(Datas.TrainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                        features = MachineLearningTools.Regularization.MaxMinNormalisation(Datas.TrainFeatures);
                    else if (scaleAdjust && normMethod == Normalization.Standardisation)
                    { features = MachineLearningTools.Regularization.ScaleAdjust(Datas.TrainFeatures); MachineLearningTools.Regularization.Standardisation(ref features); }
                    else if (scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                    { features = MachineLearningTools.Regularization.ScaleAdjust(Datas.TrainFeatures); MachineLearningTools.Regularization.MaxMinNormalisation(ref features); }
                    else
                        features = Datas.TrainFeatures;

                    double[] theta = Theta.Clone() as double[];
                    if (resetThetas && Theta != null)
                        MachineLearningTools.GeneralTools.FillArray(ref theta, false);

                    Func<double, double> sigmoidFunction = (predict) => 1 / (1 + Math.Pow(Math.E, -predict)); //Obtenção valor para funçãod e sigmoid 1/(1+e^-hθ(x))

                    Func<double[][], double[], double[], int, double> sum = (ftrs, thts, outpts, ftrIndx) => //Obtendo valor da expresão  Σ(hθ(x[m]) - y[m]).x[m][ftrIndx] --> x[m][0] = 1 
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? sigmoidFunction(predict) - outpts[m] : (sigmoidFunction(predict) - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    };

                    

                    if (regulatization)//Do gradient descent with lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            for (int j = 0; j < newTheta.Length; j++)
                                newTheta[j] = newTheta[j] * (1 - (alpha * lambda / trainElementsCount)) - (alpha / trainElementsCount) * sum(features, theta, outputs, j);
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    else//Do gradient descent without lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            for (int j = 0; j < newTheta.Length; j++)
                                newTheta[j] -= alpha / trainElementsCount * sum(features, theta, outputs, j);
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    Theta = theta.Clone() as double[];
                }
                /// <summary>
                /// Gradient descent process using all training elements from <see cref="MLDatas.TrainFeatures"/>. 
                /// </summary>
                /// <param name="interactions">Interacions quantity to minimize <see cref="Theta"/> propriety</param>
                /// <param name="scaleAdjust">Adjust each element from a <see cref="double"/>[] array dividing by max value from this array usign the following expression:<para>array[i] = array[i]/ max(array)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* max(array) -> Max value founded on "array" array</para></para></param>
                /// <param name="normMethod">Adjust each element from a <see cref="double"/>[] by:<para><see cref="Normalization.MaxMin_Normalization"/> -> Subtracting the value by the mean and dividing by the result from subtraction between maximum value and minimum value usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ (maximum - minimum)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para> maximum/ minimum -> Maximum/ Minimum value between each features from "array"</para></para> <para><see cref="Normalization.Standardisation"/> -> Subtracting the value by the mean and dividing by the standard deviation usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ √({m|i=0} Σ(array[i] - average(array))²/(m - 1))</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para>* m -> "array" elements count</para></para></param>
                /// <param name="regulatization">Apply lambda regularization constant on gradient descent process</param>
                /// <param name="resetThetas">true - Set the 0 value for each <see cref="Theta"/> array element<para>false - use the currently values for <see cref="Theta"/></para></param>
                public void CUDADoBatchGradientDescent(uint interactions, bool scaleAdjust = true, Normalization normMethod = Normalization.MaxMin_Normalization, bool regulatization = true, bool resetThetas = true)
                {
                    double trainElementsCount = Datas.TrainElementsCount;
                    double[] outputs = MachineLearningTools.GeneralTools.CUDAGetSpecificElements(0, Datas.TrainOutputs);

                    double
                        alpha = Alpha,
                        lambda = Lambda;

                    double[][] features;
                    if (scaleAdjust && normMethod == Normalization.None)
                        features = MachineLearningTools.Regularization.CUDAScaleAdjust(Datas.TrainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.Standardisation)
                        features = MachineLearningTools.Regularization.CUDAStandardisation(Datas.TrainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                        features = MachineLearningTools.Regularization.CUDAMaxMinNormalisation(Datas.TrainFeatures);
                    else if (scaleAdjust && normMethod == Normalization.Standardisation)
                    { features = MachineLearningTools.Regularization.CUDAScaleAdjust(Datas.TrainFeatures); MachineLearningTools.Regularization.CUDAStandardisation(ref features); }
                    else if (scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                    { features = MachineLearningTools.Regularization.CUDAScaleAdjust(Datas.TrainFeatures); MachineLearningTools.Regularization.CUDAMaxMinNormalisation(ref features); }
                    else
                        features = Datas.TrainFeatures;

                    double[] theta = Theta.Clone() as double[];
                    if (resetThetas && Theta != null)
                        MachineLearningTools.GeneralTools.CUDAFillArray(ref theta, false);

                    Func<double, double> sigmoidFunction = (predict) => 1 / (1 + Math.Pow(Math.E, -predict)); //Obtenção valor para funçãod e sigmoid 1/(1+e^-hθ(x))
                    Func<double[][], double[], double[], int, double> sum = (ftrs, thts, outpts, ftrIndx) => // Obtendo valor da expresão  Σ(hθ(x[m]) - y[m]).x[m][ftrIndx] --> x[m][0] = 1 
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? sigmoidFunction(predict) - outpts[m] : (sigmoidFunction(predict) - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    };


                    if (regulatization)//Do gradient descent with lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            Parallel.For(0, newTheta.Length, j => { newTheta[j] = newTheta[j] * (1 - (alpha * lambda / trainElementsCount)) - (alpha / trainElementsCount) * sum(features, theta, outputs, j); });
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    else//Do gradient descent without lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            Parallel.For(0, newTheta.Length, j => { newTheta[j] -= alpha / trainElementsCount * sum(features, theta, outputs, j); });
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    Theta = theta.Clone() as double[];
                }

                /// <summary>
                /// Gradient descent process. 
                /// </summary>
                /// <param name="trainFeatures">Train array</param>
                /// <param name="trainOutputs">Train outputs array</param>
                /// <param name="alpha">Learning rate constant
                /// <para>Small value -> More slow to reach global minimum (if too small, it's possible can't reach the global minimun)</para>
                /// <para>High value -> More fast to reach global minimum (if too high, it's possible surpass the global minimun)</para></param>
                /// <param name="lambda">Constant for gradient descent regularization</param>                
                /// <param name="interactions">Interacions quantity to minimize <see cref="Theta"/> propriety</param>
                /// <param name="scaleAdjust">Adjust each element from a <see cref="double"/>[] array dividing by max value from this array usign the following expression:<para>array[i] = array[i]/ max(array)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* max(array) -> Max value founded on "array" array</para></para></param>
                /// <param name="normMethod">Adjust each element from a <see cref="double"/>[] by:<para><see cref="Normalization.MaxMin_Normalization"/> -> Subtracting the value by the mean and dividing by the result from subtraction between maximum value and minimum value usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ (maximum - minimum)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para> maximum/ minimum -> Maximum/ Minimum value between each features from "array"</para></para> <para><see cref="Normalization.Standardisation"/> -> Subtracting the value by the mean and dividing by the standard deviation usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ √({m|i=0} Σ(array[i] - average(array))²/(m - 1))</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para>* m -> "array" elements count</para></para></param>
                /// <param name="regulatization">Apply lambda regularization constant on gradient descent process</param>
                /// <param name="resetThetas">true - Set the 0 value for each <see cref="Theta"/> array element<para>false - use the currently values for <see cref="Theta"/></para></param>
                /// <param name="previousTheta">Previous theta array value</param>
                public double[] BatchGradientDescent(double[][] trainFeatures, double[] trainOutputs, double alpha, double lambda, uint interactions, bool scaleAdjust = true, Normalization normMethod = Normalization.MaxMin_Normalization, bool regulatization = true, bool resetThetas = true, double[] previousTheta = null)
                {
                    double trainElementsCount = trainFeatures.Length;                  
                    
                    double[] outputs = trainOutputs.Clone() as double[];
                    double[] theta = null;
                    if (previousTheta != null)
                        theta = previousTheta.Clone() as double[];
                    else
                    {
                        theta = new double[trainFeatures[0].Length + 1];
                        MachineLearningTools.GeneralTools.FillArray(ref theta);
                    }

                    double[][] features;
                    if (scaleAdjust && normMethod == Normalization.None)
                        features = MachineLearningTools.Regularization.ScaleAdjust(trainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.Standardisation)
                        features = MachineLearningTools.Regularization.Standardisation(trainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                        features = MachineLearningTools.Regularization.MaxMinNormalisation(trainFeatures);
                    else if (scaleAdjust && normMethod == Normalization.Standardisation)
                    { features = MachineLearningTools.Regularization.ScaleAdjust(trainFeatures); MachineLearningTools.Regularization.Standardisation(ref features); }
                    else if (scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                    { features = MachineLearningTools.Regularization.ScaleAdjust(trainFeatures); MachineLearningTools.Regularization.MaxMinNormalisation(ref features); }
                    else
                        features = trainFeatures;
                    
                    Func<double, double> sigmoidFunction = (predict) => 1 / (1 + Math.Pow(Math.E, -predict)); //Obtenção valor para funçãod e sigmoid 1/(1+e^-hθ(x))

                    Func<double[][], double[], double[], int, double> sum = (ftrs, thts, outpts, ftrIndx) => //Obtendo valor da expresão  Σ(hθ(x[m]) - y[m]).x[m][ftrIndx] --> x[m][0] = 1 
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? sigmoidFunction(predict) - outpts[m] : (sigmoidFunction(predict) - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    };



                    if (regulatization)//Do gradient descent with lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            for (int j = 0; j < newTheta.Length; j++)
                                newTheta[j] = newTheta[j] * (1 - (alpha * lambda / trainElementsCount)) - (alpha / trainElementsCount) * sum(features, theta, outputs, j);
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    else//Do gradient descent without lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            for (int j = 0; j < newTheta.Length; j++)
                                newTheta[j] -= alpha / trainElementsCount * sum(features, theta, outputs, j);
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }

                    return theta;
                }
                /// <summary>
                /// Gradient descent process. 
                /// </summary>
                /// <param name="trainFeatures">Train array</param>
                /// <param name="trainOutputs">Train outputs array</param>
                /// <param name="alpha">Learning rate constant
                /// <para>Small value -> More slow to reach global minimum (if too small, it's possible can't reach the global minimun)</para>
                /// <para>High value -> More fast to reach global minimum (if too high, it's possible surpass the global minimun)</para></param>
                /// <param name="lambda">Constant for gradient descent regularization</param>                
                /// <param name="interactions">Interacions quantity to minimize <see cref="Theta"/> propriety</param>
                /// <param name="scaleAdjust">Adjust each element from a <see cref="double"/>[] array dividing by max value from this array usign the following expression:<para>array[i] = array[i]/ max(array)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* max(array) -> Max value founded on "array" array</para></para></param>
                /// <param name="normMethod">Adjust each element from a <see cref="double"/>[] by:<para><see cref="Normalization.MaxMin_Normalization"/> -> Subtracting the value by the mean and dividing by the result from subtraction between maximum value and minimum value usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ (maximum - minimum)</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para> maximum/ minimum -> Maximum/ Minimum value between each features from "array"</para></para> <para><see cref="Normalization.Standardisation"/> -> Subtracting the value by the mean and dividing by the standard deviation usign the following expression:</para><para>array[i] = {m|i=0} Σ(array[i] - average(array))/ √({m|i=0} Σ(array[i] - average(array))²/(m - 1))</para><para>Where:<para>* array[i] -> Element from "array" array on "i" index</para><para>* average(array) -> Average value on "array" array</para><para>* m -> "array" elements count</para></para></param>
                /// <param name="regulatization">Apply lambda regularization constant on gradient descent process</param>
                /// <param name="resetThetas">true - Set the 0 value for each <see cref="Theta"/> array element<para>false - use the currently values for <see cref="Theta"/></para></param>
                /// <param name="previousTheta">Previous theta array value</param>
                public double[] CUDABatchGradientDescent(double[][] trainFeatures, double[] trainOutputs, double alpha, double lambda, uint interactions, bool scaleAdjust = true, Normalization normMethod = Normalization.MaxMin_Normalization, bool regulatization = true, bool resetThetas = true, double[] previousTheta = null)
                {
                    double trainElementsCount = trainFeatures.Length;

                    double[] outputs = trainOutputs.Clone() as double[];
                    double[] theta = null;
                    if (previousTheta != null)
                        theta = previousTheta.Clone() as double[];
                    else
                    {
                        theta = new double[trainFeatures[0].Length + 1];
                        MachineLearningTools.GeneralTools.CUDAFillArray(ref theta);
                    }

                    double[][] features;
                    if (scaleAdjust && normMethod == Normalization.None)
                        features = MachineLearningTools.Regularization.CUDAScaleAdjust(trainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.Standardisation)
                        features = MachineLearningTools.Regularization.CUDAStandardisation(trainFeatures);
                    else if (!scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                        features = MachineLearningTools.Regularization.CUDAMaxMinNormalisation(trainFeatures);
                    else if (scaleAdjust && normMethod == Normalization.Standardisation)
                    { features = MachineLearningTools.Regularization.CUDAScaleAdjust(trainFeatures); MachineLearningTools.Regularization.CUDAStandardisation(ref features); }
                    else if (scaleAdjust && normMethod == Normalization.MaxMin_Normalization)
                    { features = MachineLearningTools.Regularization.CUDAScaleAdjust(trainFeatures); MachineLearningTools.Regularization.CUDAMaxMinNormalisation(ref features); }
                    else
                        features = trainFeatures;

                    Func<double, double> sigmoidFunction = (predict) => 1 / (1 + Math.Pow(Math.E, -predict)); //Obtenção valor para funçãod e sigmoid 1/(1+e^-hθ(x))
                    Func<double[][], double[], double[], int, double> sum = (ftrs, thts, outpts, ftrIndx) => // Obtendo valor da expresão  Σ(hθ(x[m]) - y[m]).x[m][ftrIndx] --> x[m][0] = 1 
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? sigmoidFunction(predict) - outpts[m] : (sigmoidFunction(predict) - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    };


                    if (regulatization)//Do gradient descent with lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            Parallel.For(0, newTheta.Length, j => { newTheta[j] = newTheta[j] * (1 - (alpha * lambda / trainElementsCount)) - (alpha / trainElementsCount) * sum(features, theta, outputs, j); });
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }
                    else//Do gradient descent without lambda
                    {
                        double[] newTheta = theta.Clone() as double[];
                        for (int i = 0; i < interactions; i++)
                        {
                            Parallel.For(0, newTheta.Length, j => { newTheta[j] -= alpha / trainElementsCount * sum(features, theta, outputs, j); });
                            theta = newTheta.Clone() as double[];
                            //Console.WriteLine(gt.WriteArray(theta));
                        }
                    }

                    return theta;
                }

                /// <summary>
                /// Normalizations methods
                /// </summary>
                public enum Normalization
                {
                    /// <summary>
                    /// MaxMinNormalisation or CUDAMaxMinNormalisation methods
                    /// </summary>
                    MaxMin_Normalization,
                    /// <summary>
                    /// Standardisation or CUDAStandardisation methods
                    /// </summary>
                    Standardisation,
                    /// <summary>
                    /// 
                    /// </summary>
                    None
                }
            }


            /// <summary>
            /// Neural networks class
            /// </summary>
            public class NeuralNetworks
            {
                /// <summary>
                /// Neural networks datas
                /// </summary>
                public MLDatas Datas { get; private set; }
                /// <summary>
                /// Upper thetas array
                /// </summary>
                public double[][] UpperThetas { get; private set; }
                /// <summary>
                /// Activation unit array
                /// </summary>
                public double[][] ActivationUnits { get; private set; }
                /// <summary>
                /// Hidden layers count
                /// </summary>
                public uint HiddenLayers { get; set; }

                
            }
            /// <summary>
            /// Suport vector machine class (SVM's)
            /// </summary>
            public class SupportVectorMachine
            {

            }
        }
        /// <summary>
        /// Unsupervised learning
        /// </summary>
        public static class UnsupervisedLearning
        {
            /// <summary>
            /// K-Means class
            /// </summary>
            public class KMeans
            {

            }
            /// <summary>
            /// Principal component analysis class (PCA)
            /// </summary>
            public class PrincipalComponentAnalysis
            {

            }
            /// <summary>
            /// Anomaly detection class
            /// </summary>
            public class AnomalyDetection
            {

            }
        }
        /// <summary>
        /// Special applications collection
        /// </summary>
        public static class SpecialApplications
        {
            /// <summary>
            /// Recommender systems class
            /// </summary>
            public class RecommenderSystems
            {

            }
            /// <summary>
            /// Large scale machine learning
            /// </summary>
            public class LargeScaleMachineLearning
            {

            }
        }
        /// <summary>
        /// Machine learning tools collection
        /// </summary>
        public static class MachineLearningTools
        {
            /// <summary>
            /// Regularization class
            /// </summary>
            public static class Regularization
            {   
                #region Scale Adjust
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array dividing by max value from this array
                /// </summary>
                /// <param name="array">{x0, x1, x2,..., xm}<para>m - "array" lenght</para></param>
                /// <returns></returns>
                public static double[] ScaleAdjust(double[] array)
                {
                    double[] adjustedArray = new double[array.Length];                    
                    double max = array.Max();
                    
                    for (int i = 0; i < adjustedArray.Length; i++)
                        adjustedArray[i] = array[i] / max;

                    return adjustedArray;
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array dividing by max value from this array
                /// </summary>
                /// <param name="array">{x0, x1, x2,..., xm}<para>m - "array" lenght</para></param>
                /// <returns></returns>
                public static double[] CUDAScaleAdjust(double[] array)
                {
                    double[] adjustedArray = new double[array.Length];
                    double max = array.Max();

                    Parallel.For(0, adjustedArray.Length, i => adjustedArray[i] = array[i] / max);

                    return adjustedArray;
                }

                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array dividing by max value from this array
                /// </summary>
                /// <param name="array">{x0, x1, x2,..., xm}<para>m - "array" lenght</para></param>
                /// <returns></returns>
                public static void ScaleAdjust(ref double[] array)
                {                    
                    double max = array.Max();

                    for (int i = 0; i < array.Length; i++)
                        array[i] = array[i] / max;
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array dividing by max value from this array
                /// </summary>
                /// <param name="array">{x0, x1, x2,..., xm}<para>m - "array" lenght</para></param>
                /// <returns></returns>
                public static void CUDAScaleAdjust(ref double[] array)
                {
                    double[] adjustedArray = array;
                    double max = adjustedArray.Max();

                    Parallel.For(0, adjustedArray.Length, i => adjustedArray[i] = adjustedArray[i] / max);
                }






                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array dividing by max value from this array
                /// </summary>
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static double[][] ScaleAdjust(double[][] array)
                {
                    double[][] adjustedArray = new double[array.Length][];                                        
                    double[] maxs = new double[array.First().Length];

                    GeneralTools.FillArray(ref maxs, false);

                    for (int i = 0; i < array.Length; i++)
                        for (int j = 0; j < array[i].Length; j++)
                            if (maxs[j] < array[i][j])
                                maxs[j] = array[i][j];

                    for (int i = 0; i < array.Length; i++)
                    {
                        adjustedArray[i] = new double[array[i].Length];
                        for (int j = 0; j < array[i].Length; j++)
                            adjustedArray[i][j] = array[i][j] / maxs[j];
                    }
                            
                    return adjustedArray;
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array dividing by max value from this array
                /// </summary>
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static double[][] CUDAScaleAdjust(double[][] array)
                {
                    double[][] adjustedArray = new double[array.Length][];

                    double[] maxs = new double[array.First().Length];

                    GeneralTools.CUDAFillArray(ref maxs, false);

                    Parallel.For(0, array.Length, i => { Parallel.For(0, array[i].Length, j => { if (maxs[j] < array[i][j]) maxs[j] = array[i][j]; }); });

                    Parallel.For(0, array.Length, i => { adjustedArray[i] = new double[array[i].Length]; Parallel.For(0, array[i].Length, j => { adjustedArray[i][j] = array[i][j] / maxs[j]; }); });

                    return adjustedArray;
                }

                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array dividing by max value from this array
                /// </summary>
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static double[][] ScaleAdjust(ref double[][] array)
                {
                    double[] maxs = new double[array.First().Length];

                    GeneralTools.FillArray(ref maxs, false);

                    for (int i = 0; i < array.Length; i++)
                        for (int j = 0; j < array[i].Length; j++)
                            if (maxs[j] < array[i][j])
                                maxs[j] = array[i][j];

                    for (int i = 0; i < array.Length; i++)
                        for (int j = 0; j < array[i].Length; j++)
                            array[i][j] = array[i][j] / maxs[j];

                    return array;
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array dividing by max value from this array
                /// </summary>
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static void CUDAScaleAdjust(ref double[][] array)
                {
                    double[][] adjustedArray = array;

                    double[] maxs = new double[adjustedArray.First().Length];

                    GeneralTools.CUDAFillArray(ref maxs, false);

                    Parallel.For(0, array.Length, i => { Parallel.For(0, adjustedArray[i].Length, j => { if (maxs[j] < adjustedArray[i][j]) maxs[j] = adjustedArray[i][j]; }); });

                    Parallel.For(0, adjustedArray.Length, i => { Parallel.For(0, adjustedArray[i].Length, j => { adjustedArray[i][j] = adjustedArray[i][j] / maxs[j]; }); });
                }
                #endregion

                #region Standardisation
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the standard deviation
                /// </summary>
                /// <param name="array"></param>
                /// <returns></returns>
                public static double[] Standardisation(double[] array)
                {
                    double[] normalizedArray = array.Clone() as double[];
                    double average = array.Average();                    

                    double standartDeviation = StandartDeviation();
                    for (int i = 0; i < normalizedArray.Length; i++)
                        normalizedArray[i] = (normalizedArray[i] - average) / standartDeviation;
                    return normalizedArray;

                    double StandartDeviation()
                    {
                        double sum = 0;
                        for (int i = 0; i < normalizedArray.Length; i++)
                            sum += Math.Pow((normalizedArray[i] - average), 2);
                        double sD = Math.Pow(sum / (normalizedArray.Length - 1), 0.5);
                        return sD;
                    }
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the standard deviation
                /// </summary>
                /// <param name="array"></param>
                /// <returns></returns>
                public static double[] CUDAStandardisation(double[] array)
                {
                    double[] normalizedArray = new double[array.Length];
                    double average = array.Average();
                    Parallel.For(0, normalizedArray.Length, i => normalizedArray[i] = array[i]);
                     
                    double standartDeviation = CUDAStandartDeviation();

                    Parallel.For(0, normalizedArray.Length, i => normalizedArray[i] = (normalizedArray[i] - average) / standartDeviation);

                    return normalizedArray;

                    double CUDAStandartDeviation()
                    {
                        double sum = 0;
                        double sD = 0;
                        Parallel.For(0, normalizedArray.Length, i => sum += Math.Pow((normalizedArray[i] - average), 2));
                        sD = Math.Pow(sum / (normalizedArray.Length - 1), 0.5);
                        return sD;
                    }
                }

                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the standard deviation
                /// </summary>
                /// <param name="array"></param>
                /// <returns></returns>
                public static void Standardisation(ref double[] array)
                {
                    double average = array.Average();
                    double standartDeviation = StandartDeviation(array);
                    for (int i = 0; i < array.Length; i++)
                        array[i] = (array[i] - average) / standartDeviation;

                    double StandartDeviation(double[] arr)
                    {
                        double sum = 0;
                        for (int i = 0; i < arr.Length; i++)
                            sum += Math.Pow((arr[i] - average), 2);
                        double sD = Math.Pow(sum / (arr.Length - 1), 0.5);
                        return sD;
                    }
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the standard deviation
                /// </summary>
                /// <param name="array"></param>
                /// <returns></returns>
                public static void CUDAStandardisation(ref double[] array)
                {
                    double[] normalizedArray = array;
                    double average = normalizedArray.Average();
                    double standartDeviation = CUDAStandartDeviation();

                    Parallel.For(0, normalizedArray.Length, i => normalizedArray[i] = (normalizedArray[i] - average) / standartDeviation);                    

                    double CUDAStandartDeviation()
                    {
                        double sum = 0;
                        double sD = 0;
                        Parallel.For(0, normalizedArray.Length, i => sum += Math.Pow((normalizedArray[i] - average), 2));
                        sD = Math.Pow(sum / (normalizedArray.Length - 1), 0.5);
                        return sD;
                    }
                }






                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the standard deviation
                /// </summary>                
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static double[][] Standardisation(double[][] array)
                {                    
                    double[][] normalizedArray = new double[array.Length][] ;
                    int elementsCount = array.Length;
                    int lenght = array.First().Length;

                    double[] averages = Averages();
                    double[] standartDeviations = StandartDeviations();
                    for (int i = 0; i < elementsCount; i++)
                    {
                        normalizedArray[i] = new double[array[i].Length];
                        for (int j = 0; j < lenght; j++)
                            normalizedArray[i][j] = (array[i][j] - averages[j]) / standartDeviations[j];
                    }
                    return normalizedArray;

                    double[] StandartDeviations()
                    {
                        double[] sums = new double[lenght];
                        double[] sDs = new double[lenght];

                        for (int i = 0; i < elementsCount; i++)
                            for (int j = 0; j < lenght; j++)
                                sums[j] += Math.Pow((array[i][j] - averages[j]), 2);

                        for (int i = 0; i < lenght; i++)
                            sDs[i] = Math.Pow(sums[i] / (elementsCount - 1), 0.5);

                        return sDs;
                    }

                    double[] Averages()
                    {
                        double[] sums = new double[lenght];
                        double[] avrgs = new double[lenght];

                        for (int i = 0; i < elementsCount; i++)
                            for (int j = 0; j < array[i].Length; j++)                                
                                    sums[j] += array[i][j];

                        for (int i = 0; i < lenght; i++)
                            avrgs[i] = sums[i] / elementsCount;

                        return avrgs;
                    }
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the standard deviation
                /// </summary>                
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static double[][] CUDAStandardisation(double[][] array)
                {
                    double[][] normalizedArray = new double[array.Length][];
                    int elementsCount = array.Length;
                    int lenght = array.First().Length;

                    double[] averages = Averages();
                    double[] standartDeviations = StandartDeviations();

                    Parallel.For(0, elementsCount, i => { normalizedArray[i] = new double[array[i].Length]; Parallel.For(0, lenght, j => { normalizedArray[i][j] = (array[i][j] - averages[j]) / standartDeviations[j]; }); });

                    return normalizedArray;

                    double[] StandartDeviations()
                    {
                        double[] sums = new double[lenght];
                        double[] sDs = new double[lenght];

                        Parallel.For(0, elementsCount, i => { Parallel.For(0, lenght, j => { sums[j] += Math.Pow((array[i][j] - averages[j]), 2); }); });                        
                        Parallel.For(0, lenght, j => { sDs[j] = Math.Pow(sums[j] / (elementsCount - 1), 0.5); });                        

                        return sDs;
                    }

                    double[] Averages()
                    {
                        double[] sums = new double[lenght];
                        double[] avrgs = new double[lenght];
        
                        Parallel.For(0, elementsCount, i => { Parallel.For(0, lenght, j => { sums[j] += array[i][j]; }); });
                        Parallel.For(0, lenght, j => { avrgs[j] = sums[j] / elementsCount; });

                        return avrgs;
                    }
                }

                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the standard deviation
                /// </summary>                
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static void Standardisation(ref double[][] array)
                {
                    double[][] normalizedArray = array;
                    int elementsCount = array.Length;
                    int lenght = array.First().Length;

                    double[] averages = Averages();
                    double[] standartDeviations = StandartDeviations();
                    for (int i = 0; i < elementsCount; i++)
                        for (int j = 0; j < lenght; j++)
                            normalizedArray[i][j] = (normalizedArray[i][j] - averages[j]) / standartDeviations[j];

                    double[] StandartDeviations()
                    {
                        double[] sums = new double[lenght];
                        double[] sDs = new double[lenght];

                        for (int i = 0; i < elementsCount; i++)
                            for (int j = 0; j < normalizedArray[i].Length; j++)
                                sums[j] += Math.Pow((normalizedArray[i][j] - averages[j]), 2);

                        for (int i = 0; i < lenght; i++)
                            sDs[i] = Math.Pow(sums[i] / (elementsCount - 1), 0.5); ;

                        return sDs;
                    }

                    double[] Averages()
                    {
                        double[] sums = new double[lenght];
                        double[] avrgs = new double[lenght];

                        for (int i = 0; i < elementsCount; i++)
                            for (int j = 0; j < normalizedArray[i].Length; j++)
                                sums[j] += normalizedArray[i][j];

                        for (int i = 0; i < lenght; i++)
                            avrgs[i] = sums[i] / elementsCount;

                        return avrgs;
                    }
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the standard deviation
                /// </summary>                
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static void CUDAStandardisation(ref double[][] array)
                {
                    double[][] normalizedArray = array;
                    int elementsCount = array.Length;
                    int lenght = array.First().Length;

                    double[] averages = Averages();
                    double[] standartDeviations = StandartDeviations();

                    Parallel.For(0, elementsCount, i => { Parallel.For(0, lenght, j => { normalizedArray[i][j] = (normalizedArray[i][j] - averages[j]) / standartDeviations[j]; }); });

                    double[] StandartDeviations()
                    {
                        double[] sums = new double[lenght];
                        double[] sDs = new double[lenght];

                        Parallel.For(0, elementsCount, i => { Parallel.For(0, lenght, j => { sums[j] += Math.Pow((normalizedArray[i][j] - averages[j]), 2); }); });
                        Parallel.For(0, lenght, j => { sDs[j] = Math.Pow(sums[j] / (elementsCount - 1), 0.5); });

                        return sDs;
                    }

                    double[] Averages()
                    {
                        double[] sums = new double[lenght];
                        double[] avrgs = new double[lenght];

                        Parallel.For(0, elementsCount, i => { Parallel.For(0, lenght, j => { sums[j] += normalizedArray[i][j]; }); });
                        Parallel.For(0, lenght, j => { avrgs[j] = sums[j] / elementsCount; });

                        return avrgs;
                    }
                }
                #endregion

                #region Max-Min Normalisation
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the subtraction between maximum and minimum values on array
                /// </summary>
                /// <param name="array"></param>
                /// <returns></returns>
                public static double[] MaxMinNormalisation(double[] array)
                {
                    double[] normalizedArray = array.Clone() as double[];
                    double average = array.Average();
                    double 
                        max = array.Max(), 
                        min = array.Min();                    
                    for (int i = 0; i < normalizedArray.Length; i++)
                        normalizedArray[i] = (normalizedArray[i] - average) / (max - min);
                    return normalizedArray;

                    
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the subtraction between maximum and minimum values on array
                /// </summary>
                /// <param name="array"></param>
                /// <returns></returns>
                public static double[] CUDAMaxMinNormalisation(double[] array)
                {
                    double[] normalizedArray = new double[array.Length];
                    double average = array.Average();
                    Parallel.For(0, normalizedArray.Length, i => normalizedArray[i] = array[i]);

                    double
                        max = array.Max(),
                        min = array.Min();
                    Parallel.For(0, normalizedArray.Length, i => normalizedArray[i] = (normalizedArray[i] - average) / (max - min));

                    return normalizedArray;
                }

                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the subtraction between maximum and minimum values on array
                /// </summary>
                /// <param name="array"></param>
                /// <returns></returns>
                public static void MaxMinNormalisation(ref double[] array)
                {
                    double average = array.Average();
                    double
                        max = array.Max(),
                        min = array.Min();                    
                    for (int i = 0; i < array.Length; i++)
                        array[i] = (array[i] - average) / (max - min);

                    
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[] array subtracting the value by the mean and dividing by the subtraction between maximum and minimum values on array
                /// </summary>
                /// <param name="array"></param>
                /// <returns></returns>
                public static void CUDAMaxMinNormalisation(ref double[] array)
                {
                    double[] normalizedArray = array;
                    double average = normalizedArray.Average();
                    double
                        max = array.Max(),
                        min = array.Min();
                    Parallel.For(0, normalizedArray.Length, i => normalizedArray[i] = (normalizedArray[i] - average) / (max - min));

                }






                /// <summary>
                /// Adjust each element from a <see cref="double"/>[][] array subtracting the value by the mean and dividing by the subtraction between maximum and minimum values on array
                /// </summary>                
                /// <param name="array">[0]{x0,..., xn}, [1]{x0,..., xn},..., [m]{x0,..., xn}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static double[][] MaxMinNormalisation(double[][] array)
                {
                    double[][] normalizedArray = new double[array.Length][];
                    int elementsCount = array.Length;
                    int lenght = array.First().Length;

                    var res = AveragesMaxsMins();
                    double[] averages = res.Item1;                    
                    double[][] maxsMins = res.Item2;
                    
                    for (int m = 0; m < elementsCount; m++)
                    {
                        normalizedArray[m] = new double[array[m].Length];
                        for (int n = 0; n < lenght; n++)
                            normalizedArray[m][n] = (array[m][n] - averages[n]) / (maxsMins[n][0] - maxsMins[n][1]);
                    }
                    return normalizedArray;

                    

                    Tuple<double[], double[][]> AveragesMaxsMins()
                    {
                        double[] sums = new double[lenght];
                        double[] avrgs = new double[lenght];
                        double[][] mM = new double[lenght][];

                        for (int m = 0; m < elementsCount; m++)
                            for (int n = 0; n < array[m].Length; n++)
                            {
                                if (mM[n] == null)
                                    mM[n] = new double[2] { double.NaN, double.NaN };
                                if(double.IsNaN(mM[n][0]) || array[m][n] > mM[n][0])
                                    mM[n][0] = array[m][n];
                                if (double.IsNaN(mM[n][1]) || array[m][n] < mM[n][1])
                                    mM[n][1] = array[m][n];
                                sums[n] += array[m][n];
                            }

                        for (int j = 0; j < lenght; j++)
                            avrgs[j] = sums[j] / elementsCount;

                        return new Tuple<double[], double[][]>(avrgs, mM);
                    }

                    
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[][] array subtracting the value by the mean and dividing by the subtraction between maximum and minimum values on array
                /// </summary>                
                /// <param name="array">[0]{x0,..., xn}, [1]{x0,..., xn},..., [m]{x0,..., xn}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static double[][] CUDAMaxMinNormalisation(double[][] array)
                {
                    double[][] normalizedArray = new double[array.Length][];
                    int elementsCount = array.Length;
                    int lenght = array.First().Length;

                    var res = AveragesMaxsMins();
                    double[] averages = res.Item1;
                    double[][] maxsMins = res.Item2;

                    Parallel.For(0, elementsCount, i => { normalizedArray[i] = new double[array[i].Length]; Parallel.For(0, lenght, j => { normalizedArray[i][j] = (array[i][j] - averages[j]) / (maxsMins[j][0] - maxsMins[j][1]); }); });

                    return normalizedArray;

                    Tuple<double[], double[][]> AveragesMaxsMins()
                    {
                        double[] sums = new double[lenght];
                        double[] avrgs = new double[lenght];
                        double[][] mM = new double[lenght][];
                        double[][] reArray = new double[elementsCount][];
                        Parallel.For(0, elementsCount, i => { if (reArray[i] == null) reArray[i] = new double[lenght]; Parallel.For(0, lenght, j => { reArray[i][j] = array[i][j]; }); });
                        Parallel.For(0, elementsCount, i => { Parallel.For(0, lenght, j => { if (mM[j] == null) mM[j] = new double[2]; sums[j] += array[i][j]; mM[j][0] = reArray[i].Max(); mM[j][1] = reArray[i].Min(); }); });
                        Parallel.For(0, lenght, j => { avrgs[j] = sums[j] / elementsCount; });

                        return new Tuple<double[], double[][]>(avrgs, mM);
                    }
                }

                /// <summary>
                /// Adjust each element from a <see cref="double"/>[][] array subtracting the value by the mean and dividing by the subtraction between maximum and minimum values on array
                /// </summary>                
                /// <param name="array">[0]{x0,..., xn}, [1]{x0,..., xn},..., [m]{x0,..., xn}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static void MaxMinNormalisation(ref double[][] array)
                {
                    double[][] normalizedArray = array;
                    int elementsCount = array.Length;
                    int lenght = array.First().Length;

                    var res = AveragesMaxsMins();
                    double[] averages = res.Item1;
                    double[][] maxsMins = res.Item2;

                    for (int i = 0; i < elementsCount; i++)
                        for (int j = 0; j < lenght; j++)
                            normalizedArray[i][j] = (normalizedArray[i][j] - averages[j]) / (maxsMins[j][0] - maxsMins[j][1]);

                    Tuple<double[], double[][]> AveragesMaxsMins()
                    {
                        double[] sums = new double[lenght];
                        double[] avrgs = new double[lenght];
                        double[][] mM = new double[lenght][];

                        for (int i = 0; i < elementsCount; i++)
                            for (int j = 0; j < normalizedArray[i].Length; j++)
                            {
                                if(mM[j] == null)
                                    mM[j] = new double[2] {double.NaN, double.NaN };
                                sums[j] += normalizedArray[i][j];
                                if (double.IsNaN(mM[j][0]) || mM[j][0] < normalizedArray[i][j])
                                    mM[j][0] = normalizedArray[i][j];
                                if (double.IsNaN(mM[j][1]) || mM[j][1] > normalizedArray[i][j])
                                    mM[j][1] = normalizedArray[i][j];
                            }

                        for (int i = 0; i < lenght; i++)
                            avrgs[i] = sums[i] / elementsCount;

                        return new Tuple<double[], double[][]>(avrgs, mM);
                    }
                }
                /// <summary>
                /// Adjust each element from a <see cref="double"/>[][] array subtracting the value by the mean and dividing by the subtraction between maximum and minimum values on array
                /// </summary>                
                /// <param name="array">[0]{x0,..., xm}, [1]{x0,..., xm},..., [n]{x0,..., xm}
                /// <para>m - "array" elements lenght</para>
                /// <para>n - "array" lenght</para></param>
                /// <returns></returns>
                public static void CUDAMaxMinNormalisation(ref double[][] array)
                {
                    double[][] normalizedArray = array;
                    int elementsCount = array.Length;
                    int lenght = array.First().Length;

                    var res = AveragesMaxsMins();
                    double[] averages = res.Item1;
                    double[][] maxsMins = res.Item2;

                    Parallel.For(0, elementsCount, i => { Parallel.For(0, lenght, j => { normalizedArray[i][j] = (normalizedArray[i][j] - averages[j]) / (maxsMins[j][0] - maxsMins[j][1]); }); });
                                        
                    Tuple<double[], double[][]> AveragesMaxsMins()
                    {
                        double[] sums = new double[lenght];
                        double[] avrgs = new double[lenght];
                        double[][] mM = new double[lenght][];
                        double[][] reArray = new double[elementsCount][];
                        Parallel.For(0, elementsCount, i => { if (reArray[i] == null) reArray[i] = new double[lenght]; Parallel.For(0, lenght, j => { reArray[i][j] = normalizedArray[i][j]; }); });
                        Parallel.For(0, elementsCount, i => { Parallel.For(0, lenght, j => { if (mM[j] == null) mM[j] = new double[2]; sums[j] += normalizedArray[i][j]; mM[j][0] = reArray[i].Max(); mM[j][1] = reArray[i].Min(); }); });
                        Parallel.For(0, lenght, j => { avrgs[j] = sums[j] / elementsCount; });

                        return new Tuple<double[], double[][]>(avrgs, mM);
                    }
                }
                #endregion                
            }
            /// <summary>
            /// Bias variance class
            /// </summary>
            public static class BiasVariance
            {

            }
            /// <summary>
            /// Evaluation collection
            /// </summary>
            public static class Evaluation
            {
                /// <summary>
                /// Learning algorithms class
                /// </summary>
                public static class LearningAlgorithms
                {

                }
                /// <summary>
                /// Learning curves class
                /// </summary>
                public static class LearningCurves
                {

                }
                /// <summary>
                /// Error analysis class
                /// </summary>
                public static class ErrorAnalysis
                {

                }
                /// <summary>
                /// Ceiling analysis class
                /// </summary>
                public static class CeilingAnalysis
                {

                }
            }
            /// <summary>
            /// General tools collection
            /// </summary>
            public static class GeneralTools
            {
                #region Fill Array
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static void FillArray(ref double[] array, bool setRandomValues = false)
                {
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < array.Length; i++)
                            array[i] = rnd.NextDouble();
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                            array[i] = 0;
                    }
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static void FillArray(ref double[][] array, bool setRandomValues = false)
                {
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < array.Length; i++)
                            for (int j = 0; i < array[i].Length; i++)
                                array[i][j] = rnd.NextDouble();
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                            for (int j = 0; i < array[i].Length; i++)
                                array[i][j] = 0;
                    }
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static void FillArray(ref double[][][] array, bool setRandomValues = false)
                {
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < array.Length; i++)
                            for (int j = 0; i < array[i].Length; i++)
                                for (int k = 0; i < array[i][j].Length; i++)
                                    array[i][j][k] = rnd.NextDouble();
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                            for (int j = 0; i < array[i].Length; i++)
                                for (int k = 0; i < array[i][j].Length; i++)
                                    array[i][j][k] = 0;
                    }
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static void CUDAFillArray(ref double[] array, bool setRandomValues = false)
                {
                    double[] filleArray = array;
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, array.Length, i => { filleArray[i] = rnd.NextDouble(); });
                    }
                    else
                        Parallel.For(0, array.Length, i => { filleArray[i] = 0; });
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static void CUDAFillArray(ref double[][] array, bool setRandomValues = false)
                {
                    double[][] filleArray = array;
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, array.Length, i => { Parallel.For(0, filleArray[i].Length, j => { filleArray[i][j] = rnd.NextDouble(); }); });
                    }
                    else
                        Parallel.For(0, array.Length, i => { Parallel.For(0, filleArray[i].Length, j => { filleArray[i][j] = 0; }); });
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static void CUDAFillArray(ref double[][][] array, bool setRandomValues = false)
                {
                    double[][][] filleArray = array;
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, array.Length, i => { Parallel.For(0, filleArray[i].Length, j => { Parallel.For(0, filleArray[i][j].Length, k => { filleArray[i][j][k] = rnd.NextDouble(); }); }); });
                    }
                    else
                        Parallel.For(0, array.Length, i => { Parallel.For(0, filleArray[i].Length, j => { Parallel.For(0, filleArray[i][j].Length, k => { filleArray[i][j][k] = 0; }); }); });
                }



                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[] FillArray(double[] array, bool setRandomValues = false)
                {
                    double[] newArray = new double[array.Length];

                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < array.Length; i++)
                            newArray[i] = rnd.NextDouble();
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                            newArray[i] = 0;
                    }
                    return newArray;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] FillArray(double[][] array, bool setRandomValues = false)
                {
                    double[][] newArray = new double[array.Length][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < array.Length; i++)
                        {
                            newArray[i] = new double[array[i].Length];
                            for (int j = 0; i < array[i].Length; i++)
                                newArray[i][j] = rnd.NextDouble();
                        }
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                        {
                            newArray[i] = new double[array[i].Length];
                            for (int j = 0; i < array[i].Length; i++)
                                newArray[i][j] = 0;
                        }
                    }
                    return newArray;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] FillArray(double[][][] array, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < array.Length; i++)
                        {
                            newArray[i] = new double[array[i].Length][];
                            for (int j = 0; i < array[i].Length; i++)
                            {
                                newArray[i][j] = new double[array[i][j].Length];
                                for (int k = 0; i < array[i][j].Length; i++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                        {
                            newArray[i] = new double[array[i].Length][];
                            for (int j = 0; i < array[i].Length; i++)
                            {
                                newArray[i][j] = new double[array[i][j].Length];
                                for (int k = 0; i < array[i][j].Length; i++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return array;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[] CUDAFillArray(double[] array, bool setRandomValues = false)
                {
                    double[] newArray = new double[array.Length];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, array.Length, i => { newArray[i] = rnd.NextDouble(); });
                    }
                    else
                        Parallel.For(0, array.Length, i => { newArray[i] = 0; });
                    return newArray;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] CUDAFillArray(double[][] array, bool setRandomValues = false)
                {
                    double[][] newArray = new double[array.Length][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, array.Length, i => { newArray[i] = new double[array[i].Length]; Parallel.For(0, array[i].Length, j => { newArray[i][j] = rnd.NextDouble(); }); });
                    }
                    else
                        Parallel.For(0, array.Length, i => { newArray[i] = new double[array[i].Length]; Parallel.For(0, array[i].Length, j => { newArray[i][j] = 0; }); });
                    return array;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CUDAFillArray(double[][][] array, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, array.Length, i => { newArray[i] = new double[array[i].Length][]; Parallel.For(0, array[i].Length, j => { newArray[i][j] = new double[array[i][j].Length]; Parallel.For(0, array[i][j].Length, k => { newArray[i][j][k] = rnd.NextDouble(); }); }); });
                    }
                    else
                        Parallel.For(0, array.Length, i => { newArray[i] = new double[array[i].Length][]; Parallel.For(0, array[i].Length, j => { newArray[i][j] = new double[array[i][j].Length]; Parallel.For(0, array[i][j].Length, k => { newArray[i][j][k] = 0; }); }); });
                    return array;
                }
                #endregion

                #region Creat Polynoums Array
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[] -> x1, x2, x3, x4, x5,..., xi</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> x1^expoent, x2^expoent,..., xn^expoent</para>
                /// <para>true -> x1, x2,..., xi, x1^expoent, x2^expoent,..., xi^expoent</para></param>
                /// <returns></returns>
                public static double[] CreatPolynoumsArray(double expoent, double[] array, bool appendArray = false)
                {
                    double[] newArray = appendArray ? new double[2 * array.Length] : new double[array.Length];
                    if (appendArray)
                    {
                        for (int i = 0; i < array.Length; i++)
                        {
                            newArray[i] = array[i];
                            newArray[array.Length + i] = Math.Pow(array[i], expoent);
                        }
                    }
                    else
                        for (int i = 0; i < array.Length; i++)
                            newArray[i] = Math.Pow(array[i], expoent);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[] -> x1, x2, x3, x4, x5,..., xi</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> x1^expoent, x2^expoent,..., xn^expoent</para>
                /// <para>true -> x1, x2,..., xi, x1^expoent, x2^expoent,..., xi^expoent</para></param>
                /// <returns></returns>
                public static double[] CUDACreatPolynoumsArray(double expoent, double[] array, bool appendArray = false)
                {
                    double[] newArray = appendArray ? new double[2 * array.Length] : new double[array.Length];
                    if (appendArray)
                        Parallel.For(0, array.Length, i => { newArray[i] = array[i]; newArray[array.Length + i] = Math.Pow(array[i], expoent); });
                    else
                        Parallel.For(0, array.Length, i => newArray[i] = Math.Pow(array[i], expoent));
                    return newArray;
                }

                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][] ->[1]{x1,..., xi}, [2]{x1,..., xi},..., [n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> [1]{x1^expoent,..., xi^expoent},...,[n]{x1^expoent,..., xi^expoent}</para>
                /// <para>true -> [1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[n]{x1,..., xi, x1^expoent,..., xi^expoent}</para></param>
                /// <returns></returns>
                public static double[][] CreatPolynoumsArray(double expoent, double[][] array, bool appendArray = false)
                {
                    double[][] newArray = new double[array.Length][];
                    for (int n = 0; n < array.Length; n++)
                        newArray[n] = CreatPolynoumsArray(expoent, array[n], appendArray);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][] ->[1]{x1,..., xi}, [2]{x1,..., xi},..., [n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> [1]{x1^expoent,..., xi^expoent},...,[n]{x1^expoent,..., xi^expoent}</para>
                /// <para>true -> [1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[n]{x1,..., xi, x1^expoent,..., xi^expoent}</para></param>
                /// <returns></returns>
                public static double[][] CUDACreatPolynoumsArray(double expoent, double[][] array, bool appendArray = false)
                {
                    double[][] newArray = new double[array.Length][];
                    Parallel.For(0, array.Length, n => newArray[n] = CUDACreatPolynoumsArray(expoent, array[n], appendArray));
                    return newArray;
                }

                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][][] -></para>
                /// <para>[1][1]{x1,..., xi},..., [1][n]{x1,..., xi}</para>
                /// <para>[2][1]{x1,..., xi},..., [2][n]{x1,..., xi}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi},..., [m][n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -></para>
                /// <para>[1][1]{x1^expoent,..., xi^expoent},...,[1][n]{x1^expoent,..., xi^expoent}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1^expoent,..., xi^expoent},...,[m][n]{x1^expoent,..., xi^expoent}</para>
                /// <para>true -> </para>
                /// <para>[1][1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[1][n]{x1,..., xi, x1^expoent,..., xi^expoent}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[m][n]{x1,..., xi, x1^expoent,..., xi^expoent}</para></param>
                /// <returns></returns>
                public static double[][][] CreatPolynoumsArray(double expoent, double[][][] array, bool appendArray = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    for (int n = 0; n < array.Length; n++)
                        newArray[n] = CreatPolynoumsArray(expoent, array[n], appendArray);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][][] -></para>
                /// <para>[1][1]{x1,..., xi},..., [1][n]{x1,..., xi}</para>
                /// <para>[2][1]{x1,..., xi},..., [2][n]{x1,..., xi}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi},..., [m][n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -></para>
                /// <para>[1][1]{x1^expoent,..., xi^expoent},...,[1][n]{x1^expoent,..., xi^expoent}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1^expoent,..., xi^expoent},...,[m][n]{x1^expoent,..., xi^expoent}</para>
                /// <para>true -> </para>
                /// <para>[1][1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[1][n]{x1,..., xi, x1^expoent,..., xi^expoent}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[m][n]{x1,..., xi, x1^expoent,..., xi^expoent}</para></param>
                /// <returns></returns>
                public static double[][][] CUDACreatPolynoumsArray(double expoent, double[][][] array, bool appendArray = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    Parallel.For(0, array.Length, n => newArray[n] = CUDACreatPolynoumsArray(expoent, array[n], appendArray));
                    return newArray;
                }




                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[] -> x1, x2, x3, x4, x5,..., xi</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p</para>
                /// <para>true -> x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p</para></param>
                /// <returns></returns>
                public static double[] CreatPolynoumsArray(double[] expoent, double[] array, bool appendArray = false)
                {
                    double[] newArray = appendArray ? new double[array.Length + array.Length * expoent.Length] : new double[array.Length * expoent.Length];
                    if (appendArray)
                        for (int i = 0; i < array.Length; i++)
                        {
                            if (i < array.Length)
                                newArray[i] = array[i];
                            for (int p = 0; p < expoent.Length; p++)
                                newArray[array.Length + expoent.Length * i + p] = Math.Pow(array[i], expoent[p]);
                        }
                    else
                        for (int i = 0; i < array.Length; i++)
                            for (int p = 0; p < expoent.Length; p++)
                                newArray[expoent.Length * i + p] = Math.Pow(array[i], expoent[p]);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[] -> x1, x2, x3, x4, x5,..., xi</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p</para>
                /// <para>true -> x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p</para></param>
                /// <returns></returns>
                public static double[] CUDACreatPolynoumsArray(double[] expoent, double[] array, bool appendArray = false)
                {
                    double[] newArray = appendArray ? new double[array.Length + array.Length * expoent.Length] : new double[array.Length * expoent.Length];
                    if (appendArray)
                        Parallel.For(0, array.Length, i => { if (i < array.Length) newArray[i] = array[i]; Parallel.For(0, expoent.Length, p => { newArray[array.Length + expoent.Length * i + p] = Math.Pow(array[i], expoent[p]); }); });
                    else
                        Parallel.For(0, array.Length, i => { Parallel.For(0, expoent.Length, p => { newArray[expoent.Length * i + p] = Math.Pow(array[i], expoent[p]); }); });
                    return newArray;
                }

                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][] ->[1]{x1,..., xi}, [2]{x1,..., xi},..., [n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> [1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>true -> [1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para></param>
                /// <returns></returns>
                public static double[][] CreatPolynoumsArray(double[] expoent, double[][] array, bool appendArray = false)
                {
                    double[][] newArray = new double[array.Length][];
                    for (int n = 0; n < array.Length; n++)
                        newArray[n] = CreatPolynoumsArray(expoent, array[n], appendArray);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][] ->[1]{x1,..., xi}, [2]{x1,..., xi},..., [n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> [1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>true -> [1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para></param>
                /// <returns></returns>
                public static double[][] CUDACreatPolynoumsArray(double[] expoent, double[][] array, bool appendArray = false)
                {
                    double[][] newArray = new double[array.Length][];
                    Parallel.For(0, array.Length, n => newArray[n] = CUDACreatPolynoumsArray(expoent, array[n], appendArray));
                    return newArray;
                }

                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][][] -></para>
                /// <para>[1][1]{x1,..., xi},..., [1][n]{x1,..., xi}</para>
                /// <para>[2][1]{x1,..., xi},..., [2][n]{x1,..., xi}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi},..., [m][n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -></para>
                /// <para>[1][1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[1][n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[m][n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>true -> </para>
                /// <para>[1][1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[1][n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[m][n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para></param>
                /// <returns></returns>
                public static double[][][] CreatPolynoumsArray(double[] expoent, double[][][] array, bool appendArray = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    for (int n = 0; n < array.Length; n++)
                        newArray[n] = CreatPolynoumsArray(expoent, array[n], appendArray);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][][] -></para>
                /// <para>[1][1]{x1,..., xi},..., [1][n]{x1,..., xi}</para>
                /// <para>[2][1]{x1,..., xi},..., [2][n]{x1,..., xi}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi},..., [m][n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -></para>
                /// <para>[1][1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[1][n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[m][n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>true -> </para>
                /// <para>[1][1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[1][n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[m][n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para></param>
                /// <returns></returns>
                public static double[][][] CUDACreatPolynoumsArray(double[] expoent, double[][][] array, bool appendArray = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    Parallel.For(0, array.Length, n => newArray[n] = CUDACreatPolynoumsArray(expoent, array[n], appendArray));
                    return newArray;
                }





                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[] -> x1, x2, x3, x4, x5,..., xi</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> x1^expoent, x2^expoent,..., xn^expoent</para>
                /// <para>true -> x1, x2,..., xi, x1^expoent, x2^expoent,..., xi^expoent</para></param>
                /// <returns></returns>
                public static double[] CreatPolynoumsArray(ExpoentFeatureAddress expoent, double[] array, bool appendArray = false)
                {
                    double[] newArray = appendArray ? new double[array.Length + 1] : new double[1];
                    if (appendArray)
                    {
                        for (int i = 0; i < array.Length; i++)
                        {
                            newArray[i] = array[i];
                            if (i == expoent.FeatureAddress)
                                newArray[newArray.Length - 1] = Math.Pow(array[i], expoent.Expoent);
                        }
                    }
                    else
                        newArray[0] = Math.Pow(array[expoent.FeatureAddress], expoent.Expoent);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[] -> x1, x2, x3, x4, x5,..., xi</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> x1^expoent, x2^expoent,..., xn^expoent</para>
                /// <para>true -> x1, x2,..., xi, x1^expoent, x2^expoent,..., xi^expoent</para></param>
                /// <returns></returns>
                public static double[] CUDACreatPolynoumsArray(ExpoentFeatureAddress expoent, double[] array, bool appendArray = false)
                {
                    double[] newArray = appendArray ? new double[array.Length + 1] : new double[1];
                    if (appendArray)
                        Parallel.For(0, array.Length, i => { newArray[i] = array[i]; if (i == expoent.FeatureAddress) newArray[newArray.Length - 1] = Math.Pow(array[i], expoent.Expoent); });
                    else
                        newArray[0] = Math.Pow(array[expoent.FeatureAddress], expoent.Expoent);
                    return newArray;
                }

                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][] ->[1]{x1,..., xi}, [2]{x1,..., xi},..., [n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> [1]{x1^expoent,..., xi^expoent},...,[n]{x1^expoent,..., xi^expoent}</para>
                /// <para>true -> [1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[n]{x1,..., xi, x1^expoent,..., xi^expoent}</para></param>
                /// <returns></returns>
                public static double[][] CreatPolynoumsArray(ExpoentFeatureAddress expoent, double[][] array, bool appendArray = false)
                {
                    double[][] newArray = new double[array.Length][];
                    for (int n = 0; n < array.Length; n++)
                        newArray[n] = CreatPolynoumsArray(expoent, array[n], appendArray);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][] ->[1]{x1,..., xi}, [2]{x1,..., xi},..., [n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> [1]{x1^expoent,..., xi^expoent},...,[n]{x1^expoent,..., xi^expoent}</para>
                /// <para>true -> [1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[n]{x1,..., xi, x1^expoent,..., xi^expoent}</para></param>
                /// <returns></returns>
                public static double[][] CUDACreatPolynoumsArray(ExpoentFeatureAddress expoent, double[][] array, bool appendArray = false)
                {
                    double[][] newArray = new double[array.Length][];
                    Parallel.For(0, array.Length, n => newArray[n] = CUDACreatPolynoumsArray(expoent, array[n], appendArray));
                    return newArray;
                }

                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][][] -></para>
                /// <para>[1][1]{x1,..., xi},..., [1][n]{x1,..., xi}</para>
                /// <para>[2][1]{x1,..., xi},..., [2][n]{x1,..., xi}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi},..., [m][n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -></para>
                /// <para>[1][1]{x1^expoent,..., xi^expoent},...,[1][n]{x1^expoent,..., xi^expoent}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1^expoent,..., xi^expoent},...,[m][n]{x1^expoent,..., xi^expoent}</para>
                /// <para>true -> </para>
                /// <para>[1][1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[1][n]{x1,..., xi, x1^expoent,..., xi^expoent}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[m][n]{x1,..., xi, x1^expoent,..., xi^expoent}</para></param>
                /// <returns></returns>
                public static double[][][] CreatPolynoumsArray(ExpoentFeatureAddress expoent, double[][][] array, bool appendArray = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    for (int n = 0; n < array.Length; n++)
                        newArray[n] = CreatPolynoumsArray(expoent, array[n], appendArray);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent</param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][][] -></para>
                /// <para>[1][1]{x1,..., xi},..., [1][n]{x1,..., xi}</para>
                /// <para>[2][1]{x1,..., xi},..., [2][n]{x1,..., xi}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi},..., [m][n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -></para>
                /// <para>[1][1]{x1^expoent,..., xi^expoent},...,[1][n]{x1^expoent,..., xi^expoent}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1^expoent,..., xi^expoent},...,[m][n]{x1^expoent,..., xi^expoent}</para>
                /// <para>true -> </para>
                /// <para>[1][1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[1][n]{x1,..., xi, x1^expoent,..., xi^expoent}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi, x1^expoent,..., xi^expoent},...,[m][n]{x1,..., xi, x1^expoent,..., xi^expoent}</para></param>
                /// <returns></returns>
                public static double[][][] CUDACreatPolynoumsArray(ExpoentFeatureAddress expoent, double[][][] array, bool appendArray = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    Parallel.For(0, array.Length, n => newArray[n] = CUDACreatPolynoumsArray(expoent, array[n], appendArray));
                    return newArray;
                }





                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[] -> x1, x2, x3, x4, x5,..., xi</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p</para>
                /// <para>true -> x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p</para></param>
                /// <returns></returns>
                public static double[] CreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[] array, bool appendArray = false)
                {
                    double[] newArray = appendArray ? new double[array.Length + expoent.Length] : new double[expoent.Length];
                    Tuple<ExpoentFeatureAddress, int>[][] expoentsAddress = new Tuple<ExpoentFeatureAddress, int>[array.Length][];
                    for (int i = 0, j = 0; i < expoentsAddress.Length; i++)
                    {
                        ExpoentFeatureAddress[] expArray = expoent.Where(exp => exp.FeatureAddress == i).ToArray();
                        for (int p = 0; p < expArray.Length; p++)
                            expoentsAddress[i] = expoentsAddress[i] == null ? new Tuple<ExpoentFeatureAddress, int>[] { new Tuple<ExpoentFeatureAddress, int>(expArray[p], j++) } : expoentsAddress[i].Append(new Tuple<ExpoentFeatureAddress, int>(expArray[p], j++)).ToArray();
                    }
                    if (appendArray)
                        for (int i = 0; i < array.Length; i++)
                        {
                            if (i < array.Length)
                                newArray[i] = array[i];
                            if (expoentsAddress[i] != null)
                                for (int p = 0; p < expoentsAddress[i].Length; p++)
                                    newArray[array.Length + expoentsAddress[i][p].Item2] = Math.Pow(array[i], expoentsAddress[i][p].Item1.Expoent);
                        }
                    else
                        for (int i = 0; i < array.Length; i++)
                            if (expoentsAddress[i] != null)
                                for (int p = 0; p < expoentsAddress[i].Length; p++)
                                    newArray[expoentsAddress[i][p].Item2] = Math.Pow(array[i], expoentsAddress[i][p].Item1.Expoent);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[] -> x1, x2, x3, x4, x5,..., xi</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p</para>
                /// <para>true -> x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p</para></param>
                /// <returns></returns>
                public static double[] CUDACreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[] array, bool appendArray = false)
                {
                    double[] newArray = appendArray ? new double[array.Length + expoent.Length] : new double[expoent.Length];
                    Tuple<ExpoentFeatureAddress, int>[][] expoentsAddress = new Tuple<ExpoentFeatureAddress, int>[array.Length][];
                    int[] expLenghts = new int[array.Length];
                    ExpoentFeatureAddress[][] exps = new ExpoentFeatureAddress[array.Length][];
                    Parallel.For(0, array.Length, i => { exps[i] = expoent.Where(exp => exp.FeatureAddress == i).ToArray(); });
                    for (int i = 0, j = 0; i < array.Length; i++)
                        if (exps[i].Length > 0)
                            for (int p = 0; p < exps[i].Length; i++)
                                expoentsAddress[i] = expoentsAddress[i] == null ? new Tuple<ExpoentFeatureAddress, int>[] { new Tuple<ExpoentFeatureAddress, int>(exps[i][p], j++) } : expoentsAddress[i].Append(new Tuple<ExpoentFeatureAddress, int>(exps[i][p], j++)).ToArray();

                    if (appendArray)
                        Parallel.For(0, array.Length, i => { if (i < array.Length) newArray[i] = array[i]; if (expoentsAddress[i] != null) Parallel.For(0, expoentsAddress[i].Length, p => { newArray[array.Length + expoentsAddress[i][p].Item2] = Math.Pow(array[i], expoentsAddress[i][p].Item1.Expoent); }); });
                    else
                        Parallel.For(0, array.Length, i => { if (expoentsAddress[i] != null) Parallel.For(0, expoentsAddress[i].Length, p => { newArray[expoentsAddress[i][p].Item2] = Math.Pow(array[i], expoentsAddress[i][p].Item1.Expoent); }); });
                    return newArray;
                }

                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][] ->[1]{x1,..., xi}, [2]{x1,..., xi},..., [n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> [1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>true -> [1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para></param>
                /// <returns></returns>
                public static double[][] CreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[][] array, bool appendArray = false)
                {
                    double[][] newArray = new double[array.Length][];
                    for (int n = 0; n < array.Length; n++)
                        newArray[n] = CreatPolynoumsArray(expoent, array[n], appendArray);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][] ->[1]{x1,..., xi}, [2]{x1,..., xi},..., [n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -> [1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>true -> [1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para></param>
                /// <returns></returns>
                public static double[][] CUDACreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[][] array, bool appendArray = false)
                {
                    double[][] newArray = new double[array.Length][];
                    Parallel.For(0, array.Length, n => newArray[n] = CUDACreatPolynoumsArray(expoent, array[n], appendArray));
                    return newArray;
                }

                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][][] -></para>
                /// <para>[1][1]{x1,..., xi},..., [1][n]{x1,..., xi}</para>
                /// <para>[2][1]{x1,..., xi},..., [2][n]{x1,..., xi}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi},..., [m][n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -></para>
                /// <para>[1][1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[1][n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[m][n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>true -> </para>
                /// <para>[1][1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[1][n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[m][n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para></param>
                /// <returns></returns>
                public static double[][][] CreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[][][] array, bool appendArray = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    for (int n = 0; n < array.Length; n++)
                        newArray[n] = CreatPolynoumsArray(expoent, array[n], appendArray);
                    return newArray;
                }
                /// <summary>
                /// Creat a new array with each value powered by a informed expoent
                /// </summary>
                /// <param name="expoent">Expoent array
                /// <para>expoent[] -> expoent 1, expoent 2, expoent 3, expoent 4, expoent 5, ..., expoent p</para></param>
                /// <param name="array">Array where this function will apply how like de following struct (where 'x' is same how like 'array' argument):
                /// <para>array[][][] -></para>
                /// <para>[1][1]{x1,..., xi},..., [1][n]{x1,..., xi}</para>
                /// <para>[2][1]{x1,..., xi},..., [2][n]{x1,..., xi}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1,..., xi},..., [m][n]{x1,..., xi}</para></param>
                /// <param name="appendArray">Append the informed array on polynoums array:
                /// <para>false -></para>
                /// <para>[1][1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[1][n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p},...,[m][n]{x1^expoent 1, ..., x1^expoent p,..., xn^expoent 1,..., xn^expoent p}</para>
                /// <para>true -> </para>
                /// <para>[1][1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[1][n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para>
                /// <para>...</para>
                /// <para>[m][1]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p},...,[m][n]{x1, x2,..., xi, x1^expoent 1,..., x1^expoent p,..., xi^expoent 1, xi^expoent p}</para></param>
                /// <returns></returns>
                public static double[][][] CUDACreatPolynoumsArray(ExpoentFeatureAddress[] expoent, double[][][] array, bool appendArray = false)
                {
                    double[][][] newArray = new double[array.Length][][];
                    Parallel.For(0, array.Length, n => newArray[n] = CUDACreatPolynoumsArray(expoent, array[n], appendArray));
                    return newArray;
                }
                #endregion

                #region Get Specific Elements
                /// <summary>
                /// Return a new array with values from a specific index inside each array from a jagged array
                /// </summary>
                /// <param name="index"></param>
                /// <param name="jaggedArray"></param>
                /// <returns></returns>
                public static double[] GetSpecificElements(int index, double[][] jaggedArray)
                {
                    double[] array = new double[jaggedArray.Length];
                    for (int i = 0; i < jaggedArray.Length; i++)
                        array[i] = jaggedArray[i][index];
                    return array;
                }

                /// <summary>
                /// Return a new array with values from a specific index inside each array from a jagged array using CUDA Hybridizer
                /// </summary>
                /// <param name="index"></param>
                /// <param name="jaggedArray"></param>
                /// <returns></returns>
                public static double[] CUDAGetSpecificElements(int index, double[][] jaggedArray)
                {
                    double[] array = new double[jaggedArray.Length];
                    Parallel.For(0, jaggedArray.Length, i => array[i] = jaggedArray[i][index]);
                    return array;
                }
                #endregion

                #region Transpose Jagged Array/ Array
                /// <summary>
                /// Transpose a jagged array
                /// <para>*Warning: Each array from jagged array must have the same lenght</para>
                /// </summary>
                /// <returns></returns>
                /// <exception cref="ArgumentException"></exception>
                public static double[][] Transpose(double[][] jaggedArray)
                {
                    if (jaggedArray.Length == 0) throw new ArgumentException("Jagged array with 0 elements");

                    int
                        tJALenght = jaggedArray[0].Length,
                        tALenght = jaggedArray.Length;

                    double[][] tJaggedArray = new double[tJALenght][];
                    for (int i = 0; i < tJALenght; i++)
                        if (tJALenght != jaggedArray[i].Length) throw new ArgumentException("Jagged array with irregular array sizes");
                        else
                        {
                            tJaggedArray[i] = new double[tALenght];
                            for (int j = 0; j < tALenght; j++)
                                tJaggedArray[i][j] = jaggedArray[j][i];
                        }

                    return tJaggedArray;
                }
                /// <summary>
                /// Transpose array and converto to a jagged array            
                /// </summary>
                /// <returns></returns>
                /// <exception cref="ArgumentException"></exception>
                public static double[][] Transpose(double[] array)
                {
                    if (array.Length == 0) throw new ArgumentException("Array with 0 elements");
                    int
                        tJALenght = array.Length;
                    double[][] tJaggedArray = new double[tJALenght][];
                    for (int i = 0; i < tJALenght; i++)
                        tJaggedArray[i] = new double[1] { array[i] };
                    return tJaggedArray;
                }

                
                /// <summary>
                /// Transpose a jagged array using CUDA Hybridizer
                /// <para>*Warning: Each array from jagged array must have the same lenght</para>
                /// </summary>
                /// <returns></returns>
                /// <exception cref="ArgumentException"></exception>
                public static double[][] CUDATranspose(double[][] jaggedArray)
                {
                    if (jaggedArray.Length == 0) throw new ArgumentException("Jagged array with 0 elements");
                    int
                        tJALenght = jaggedArray[0].Length,
                        tALenght = jaggedArray.Length;
                    double[][] tJaggedArray = new double[tJALenght][];
                    bool[] throwExcp = new bool[tJALenght];
                    Parallel.For(0, tJALenght, i => throwExcp[i] = tJALenght != jaggedArray[i].Length);
                    if (throwExcp.Any(tE => tE)) throw new ArgumentException("Jagged array with irregular array sizes");
                    Parallel.For(0, tJALenght, i => { tJaggedArray[i] = new double[tALenght]; Parallel.For(0, tALenght, j => tJaggedArray[i][j] = jaggedArray[j][i]); });
                    return tJaggedArray;
                }
                /// <summary>
                /// Transpose array and converto to a jagged array using CUDA Hybridizer            
                /// </summary>
                /// <returns></returns>
                /// <exception cref="ArgumentException"></exception>
                public static double[][] CUDATranspose(double[] array)
                {
                    if (array.Length == 0) throw new ArgumentException("Array with 0 elements");
                    int
                        tJALenght = array.Length;
                    double[][] tJaggedArray = new double[tJALenght][];
                    Parallel.For(0, tJALenght, i => tJaggedArray[i] = new double[1] { array[i] });
                    return tJaggedArray;
                }
                #endregion

                /// <summary>
                /// Expoent feature address struct
                /// </summary>
                public struct ExpoentFeatureAddress
                {
                    /// <summary>
                    /// Expoent value
                    /// </summary>
                    public double Expoent { get; private set; }
                    /// <summary>
                    /// Feature address from a specific array
                    /// </summary>
                    public int FeatureAddress { get; private set; }
                    /// <summary>
                    /// Creat a new expoent feature address
                    /// </summary>
                    /// <param name="expoent">Expoent value</param>
                    /// <param name="featureAddress">Expoent feature address</param>
                    public ExpoentFeatureAddress(double expoent, int featureAddress)
                    {
                        Expoent = expoent;
                        FeatureAddress = featureAddress;
                    }
                    /// <summary>
                    /// 
                    /// </summary>
                    /// <returns></returns>
                    public override string ToString()
                    {
                        return $"(Expoent: {Expoent};Feature Address: {FeatureAddress})";
                    }
                }
            }
        }

        /////Os tópicos abaixo deverão ser desenvolvidos, todos se baseando tanto no processo comum e lento, quanto no processo CUDA, buscando sempre utilizar a GPU e não a CPU

        ////Obs: A existência do conjunto Dev, serve para verificar em relação ao conjunto test, indicando de o valor obtido é satisfatório ou não

        /////Data spliting:
        /////
        /////-> 100/1000/10000: 
        /////     (Without Dev.)
        /////     *Train - 70%
        /////     *Test - 30%
        /////     
        /////     (With Dev.)
        /////     *Train - 60%
        /////     *Test - 20%
        /////     *Dev - 20%
        /////     
        /////-> 1000000+: 
        /////     (Without Dev.)
        /////     *Train - 98%
        /////     *Test - 2%
        /////     
        /////     (With Dev.)
        /////     *Train - 98%
        /////     *Test - 1%
        /////     *Dev - 1%
        /////     


        /////Supervised Learning:
        /////-Linear Regression
        /////-Logistic Regression
        /////-Neural Networks
        /////-SVMs *PROCESSOS QUE MELHORAM, AGILIZAM O PROCESSO DE GRADIENTE DESCENDENTE*
        /////

        /////Unsupervised Learning:
        /////-K-means
        /////-PCA
        /////-Anomaly detection
        /////

        /////Special applications/ Special Topis:
        /////-Recommender Systems
        /////-Large Scale Machine Learning
        /////

        /////Advice on Building a Machine Learning System:
        /////-Bias/ Variance
        /////-Regularization:
        /////-Deciding what to work on next:
        ///// *Evaluation of Learning Algorithms
        ///// *Learning Curves
        ///// *Error Analysis
        ///// *Ceiling Analysis
        ///// 
        ///


        public static class gt
        {

           public static string WriteArray(double[] array)
            {
                string t = "{";
                foreach (double x in array)
                    t += t == "{" ? $"{x}" : $";{x}";
                t += "}";
                return t;
            }
            public static string WriteArray(double[][] array, string previousText = "")
            {
                string t = string.Empty;
                for (int i = 0; i < array.Length; i++)
                    t += string.IsNullOrEmpty(t) ? $"{previousText}[{i}]{WriteArray(array[i])}" : $"\n{previousText}[{i}]{WriteArray(array[i])}";
                return t;
            }
            public static string WriteArray(double[][][] array)
            {
                string t = string.Empty;
                for (int i = 0; i < array.Length; i++)
                    t += string.IsNullOrEmpty(t) ? $"{WriteArray(array[i], $"[{i}]")}" : $"\n{WriteArray(array[i], $"[{i}]")}";
                return t;
            }
            public static string WriteArray(MachineLearning.MachineLearningTools.GeneralTools.ExpoentFeatureAddress[] array)
            {
                string t = "{";
                foreach (MachineLearning.MachineLearningTools.GeneralTools.ExpoentFeatureAddress x in array)
                    t += t == "{" ? $"{x}" : $";{x}";
                t += "}";
                return t;
            }
        }

    }
}


