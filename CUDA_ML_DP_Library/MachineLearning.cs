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
            /// How much classes each <see cref="Outputs"/> elements have
            /// </summary>
            public int OutputClassCount { get; protected set; } = 0;


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
                OutputClassCount = 1;
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
                OutputClassCount = Outputs.FirstOrDefault() is double[] otpts ? otpts.Length : 0;

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
                    cO = aO + bO;

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
            /// 
            /// </summary>
            public delegate void ProportionChangedEventHandler();
            /// <summary>
            /// Occours when the <see cref="SetProportions(uint, uint, uint)"/> method is called
            /// </summary>
            public event ProportionChangedEventHandler ProportionChanged;
            /// <summary>
            /// 
            /// </summary>
            protected virtual void OnProportionChanged() => ProportionChanged?.Invoke();

            /// <summary>
            /// 
            /// </summary>
            public delegate void SetsShuffledEventHandler();
            /// <summary>
            /// Occours when the <see cref="ShuffleSets()"/> method is called
            /// </summary>
            public event SetsShuffledEventHandler SetsShuffled;
            /// <summary>
            /// 
            /// </summary>
            protected virtual void OnSetsShuffled() => SetsShuffled?.Invoke();

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

                OnProportionChanged();
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
                OnSetsShuffled();
            }

            /// <summary>
            /// Clone this instance without any reference
            /// </summary>
            /// <returns></returns>
            public MLDatas WiseClone()
            {
                return (MLDatas)this.MemberwiseClone();
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
                OutputClassCount = 1;
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

                OutputClassCount = Outputs.FirstOrDefault() is double[] otpts ? otpts.Length : 0;

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
            /// <param name="outputs">Outputs array</param>
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
                /// <para>*Warning: Linear Regression class will use only the 0 index from jagged array </para>
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
                /// <para>*Warning: Linear Regression class will use only the 0 index from jagged array </para></param>
                /// <param name="alpha"></param>
                /// <param name="lambda"></param>
                /// <param name="cudaFill">Fill <see cref="Theta"/> array using <see cref="MachineLearningTools.GeneralTools.CUDAFillArray(double[], bool)"/></param>
                public LinearRegression(MLDatas datas, double alpha, double lambda, bool cudaFill)
                {
                    Datas = datas.WiseClone();
                    Alpha = alpha;
                    Lambda = lambda;

                    int ftrsLenght = datas.FeaturesCount + 1;/*recursos + constante unitário*/
                    if (cudaFill)
                        Theta = MachineLearningTools.GeneralTools.CUDACreatFillArray(ftrsLenght, false);
                    else
                        Theta = MachineLearningTools.GeneralTools.CreatFillArray(ftrsLenght, false);
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

                    double sum(double[][] ftrs, double[] thts, double[] outpts, int ftrIndx)
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
                    }

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


                    double sum(double[][] ftrs, double[] thts, double[] outpts, int ftrIndx)
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
                    }


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
                /// Return the cost function value from this instance. Using the currently <see cref="Theta"/> array values, <see cref="MLDatas.TrainFeatures"/> and <see cref="MLDatas.TrainOutputs"/> array values from <see cref="Datas"/> propriety.
                /// </summary>
                /// <returns></returns>
                public double CostFunction()
                {
                    double sumPredOut() { double[][] tF = Datas.TrainFeatures, tO = Datas.TrainOutputs; double sum = 0; for (int m = 0; m < tF.Length; m++) sum += Math.Pow((Predict(tF[m]) - tO[m][0]), 2); return sum; }; ;
                    double sumThetaSquare() { double sum = 0; for (int i = 0; i < Theta.Length; i++) sum += Math.Pow(Theta[i], 2); return sum; };

                    double elementsCount = Datas.TrainElementsCount;
                    double cost = (1 / (2 * elementsCount)) * (sumPredOut() + Lambda * sumThetaSquare());
                    return cost;
                }
                /// <summary>
                /// Return the cost function value from this instance. Using the currently <see cref="Theta"/> array values.
                /// <para>*Warning: from <paramref name="outputs"/> argument, only will be used the 0 index from each array inside the jagged array</para>
                /// </summary>
                /// <param name="features">Elements with your resources array</param>
                /// <param name="outputs">Outputs array</param>
                /// <returns></returns>
                public double CostFunction(double[][] features, double[][] outputs)
                {
                    double sumPredOut() { double[][] tF = features, tO = outputs; double sum = 0; for (int m = 0; m < tF.Length; m++) sum += Math.Pow((Predict(tF[m]) - tO[m][0]), 2); return sum; }; ;
                    double sumThetaSquare() { double sum = 0; for (int i = 0; i < Theta.Length; i++) sum += Math.Pow(Theta[i], 2); return sum; };

                    double elementsCount = Datas.TrainElementsCount;
                    double cost = (1 / (2 * elementsCount)) * (sumPredOut() + Lambda * sumThetaSquare());
                    return cost;
                }
                /// <summary>
                /// Return the cost function value.
                /// <para>*Warning: from <paramref name="outputs"/> argument, only will be used the 0 index from each array inside the jagged array</para>
                /// </summary>
                /// <param name="features">Elements with your resources array</param>
                /// <param name="outputs">Outputs array</param>
                /// <param name="theta">Theta array</param>
                /// <returns></returns>
                public double CostFunction(double[][] features, double[][] outputs, double[] theta)
                {
                    double sumPredOut() { double[][] tF = features, tO = outputs; double sum = 0; for (int m = 0; m < tF.Length; m++) sum += Math.Pow((Predict(theta, tF[m]) - tO[m][0]), 2); return sum; }; ;
                    double sumThetaSquare() { double sum = 0; for (int i = 0; i < theta.Length; i++) sum += Math.Pow(theta[i], 2); return sum; };

                    double elementsCount = Datas.TrainElementsCount;
                    double cost = (1 / (2 * elementsCount)) * (sumPredOut() + Lambda * sumThetaSquare());
                    return cost;
                }


                /// <summary>
                /// Linear regression value hypothesis
                /// </summary>
                /// <param name="resources">Resources from a element</param>
                /// <param name="theta">theta values</param>
                /// <returns>
                /// Prediction = theta[0] + theta[1] * resources[0] + theta[2] * resources[1] + ... + theta[n] * resources[n - 1]
                /// <para>Where:</para>
                /// <para>n -> Resources quantity</para>
                /// </returns>
                public double Predict(double[] theta, double[] resources)
                {
                    double prediction = theta[0];
                    for (int n = 0; n < resources.Length; n++)
                        prediction += theta[n + 1] * resources[n];
                    return prediction;
                }
                /// <summary>
                /// Linear regression value hypothesis
                /// </summary>
                /// <param name="resources">Resources from a element</param>
                /// <returns>
                /// Prediction = theta[0] + theta[1] * resources[0] + theta[2] * resources[1] + ... + theta[n] * resources[n - 1]
                /// <para>Where:</para>
                /// <para>n -> Resources quantity</para>
                /// </returns>
                public double Predict(double[] resources)
                {
                    double prediction = Theta[0];
                    for (int n = 0; n < resources.Length; n++)
                        prediction += Theta[n + 1] * resources[n];
                    return prediction;
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
                /// <para>*Warning: Logistic Regression class will use only the 0 index from jagged array </para>
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
                /// <para>*Warning²: Logistic Regression class will pleny work when <see cref="MLDatas.Outputs"/> propriety has False value, because will be used only the 0 index from jagged array </para></param>
                /// <param name="alpha"></param>
                /// <param name="lambda"></param>
                /// <param name="cudaFill">Fill <see cref="Theta"/> array using <see cref="MachineLearningTools.GeneralTools.CUDAFillArray(double[], bool)"/></param>
                public LogisticRegression(MLDatas datas, double alpha, double lambda, bool cudaFill)
                {
                    Datas = datas.WiseClone();
                    Alpha = alpha;
                    Lambda = lambda;

                    int ftrsLenght = datas.FeaturesCount + 1;/*recursos + constante unitário*/
                    if (cudaFill)
                        Theta = MachineLearningTools.GeneralTools.CUDACreatFillArray(ftrsLenght, false);
                    else
                        Theta = MachineLearningTools.GeneralTools.CreatFillArray(ftrsLenght, false);
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

                    

                    double sum(double[][] ftrs, double[] thts, double[] outpts, int ftrIndx)
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? Sigmoid(predict) - outpts[m] : (Sigmoid(predict) - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    }



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

                    
                    double sum(double[][] ftrs, double[] thts, double[] outpts, int ftrIndx)
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? Sigmoid(predict) - outpts[m] : (Sigmoid(predict) - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    }


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
                    else if (resetThetas && Theta != null)
                    {
                        theta = new double[trainFeatures[0].Length + 1];
                        MachineLearningTools.GeneralTools.FillArray(ref theta, false);
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

                    double sum(double[][] ftrs, double[] thts, double[] outpts, int ftrIndx)
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? Sigmoid(predict) - outpts[m] : (Sigmoid(predict) - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    }



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
                    else if (resetThetas && Theta != null)
                    {
                        theta = new double[trainFeatures[0].Length + 1];
                        MachineLearningTools.GeneralTools.CUDAFillArray(ref theta, false);
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

                    
                    double sum(double[][] ftrs, double[] thts, double[] outpts, int ftrIndx)
                    {
                        double sm = 0;
                        for (int m = 0; m < ftrs.Length; m++)
                        {
                            double predict = thts[0];
                            for (int n = 0; n < ftrs[m].Length; n++)
                                predict += thts[n + 1] * ftrs[m][n];
                            sm += ftrIndx == 0 ? Sigmoid(predict) - outpts[m] : (Sigmoid(predict) - outpts[m]) * ftrs[m][ftrIndx - 1];
                        }
                        return sm;
                    }


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
                /// Return the cost function value from this instance. Using the currently <see cref="Theta"/> array values, <see cref="MLDatas.TrainFeatures"/> and <see cref="MLDatas.TrainOutputs"/> array values from <see cref="Datas"/> propriety.
                /// </summary>
                /// <returns></returns>
                public double CostFunction()
                {
                    double sumCost() { double[][] tF = Datas.TrainFeatures, tO = Datas.TrainOutputs; double sum = 0; for (int m = 0; m < tF.Length; m++) { double pred = Hypothesis(tF[m]); sum += tO[m][0] * Math.Log(pred) + (1 - tO[m][0]) * Math.Log(1 - pred); } return sum; }; ;

                    double elementsCount = Datas.TrainElementsCount;
                    double cost = -(1 / elementsCount) * sumCost();
                    return cost;
                }
                /// <summary>
                /// Return the cost function value from this instance. Using the currently <see cref="Theta"/> array values.
                /// <para>*Warning: from <paramref name="outputs"/> argument, only will be used the 0 index from each array inside the jagged array</para>
                /// </summary>
                /// <param name="features">Elements with your resources array</param>
                /// <param name="outputs">Outputs array</param>
                /// <returns></returns>
                public double CostFunction(double[][] features, double[][] outputs)
                {
                    double sumCost() { double[][] tF = features, tO = outputs; double sum = 0; for (int m = 0; m < tF.Length; m++) { double pred = Hypothesis(tF[m]); sum += tO[m][0] * Math.Log(pred) + (1 - tO[m][0]) * Math.Log(1 - pred); } return sum; }; ;

                    double elementsCount = features.Length;
                    double cost = -(1 / elementsCount) * sumCost();
                    return cost;
                }
                /// <summary>
                /// Return the cost function value.
                /// <para>*Warning: from <paramref name="outputs"/> argument, only will be used the 0 index from each array inside the jagged array</para>
                /// </summary>
                /// <param name="features">Elements with your resources array</param>
                /// <param name="outputs">Outputs array</param>
                /// <param name="theta">Theta array</param>
                /// <returns></returns>
                public double CostFunction(double[][] features, double[][] outputs, double[] theta)
                {
                    double sumCost() { double[][] tF = features, tO = outputs; double sum = 0; for (int m = 0; m < tF.Length; m++) { double pred = Hypothesis(theta, tF[m]); sum += tO[m][0] * Math.Log(pred) + (1 - tO[m][0]) * Math.Log(1 - pred); } return sum; }; ;

                    double elementsCount = features.Length;
                    double cost = -(1 / elementsCount) * sumCost();
                    return cost;
                }

                /// <summary>
                /// Return the sigmoid value from a informed value
                /// </summary>
                /// <param name="value"></param>
                /// <returns></returns>
                public double Sigmoid(double value) => 1 / (1 + Math.Pow(Math.E, -value));

                /// <summary>
                /// Logistic regression value hypothesis
                /// </summary>
                /// <param name="resources">Resources from a element</param>
                /// <param name="theta">theta values</param>
                /// <returns>
                /// Prediction = theta[0] + theta[1] * resources[0] + theta[2] * resources[1] + ... + theta[n] * resources[n - 1]
                /// <para>Where:</para>
                /// <para>n -> Resources quantity</para>
                /// </returns>
                public double Hypothesis(double[] theta, double[] resources)
                {
                    double prediction = theta[0];
                    for (int n = 0; n < resources.Length; n++)
                        prediction += theta[n + 1] * resources[n];
                    return prediction;
                }
                /// <summary>
                /// Logistic regression value hypothesis
                /// </summary>
                /// <param name="resources">Resources from a element</param>
                /// <returns>
                /// Prediction = theta[0] + theta[1] * resources[0] + theta[2] * resources[1] + ... + theta[n] * resources[n - 1]
                /// <para>Where:</para>
                /// <para>n -> Resources quantity</para>
                /// </returns>
                public double Hypothesis(double[] resources)
                {
                    double prediction = Theta[0];
                    for (int n = 0; n < resources.Length; n++)
                        prediction += Theta[n + 1] * resources[n];
                    return prediction;
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
                /// Upper thetas array (double[l][a][n])
                /// <para>Where: </para>
                /// <para>l -> Layers <para>__'Layers' Lenght:<para>______l = [Hidden Layers (<see cref="HiddenLayers"/>) + Output Layer (1)]</para></para></para>
                /// <para>a -> Node (Actiovation Unit) <para>__'Node' Lenght:<para>______a = [Features Count (<see cref="MLDatas.FeaturesCount"/>) + bias unit (1)] **for <see cref="int"/> hiddenLayers constructor argument value**<para>______a = [Hidden Layers Units Count (variable) + bias unit (1)] **for <see cref="int"/>[] hiddenLayers constructor argument value**<para>__________Range: <see cref="UpperThetas"/>[0] to <see cref="UpperThetas"/>[l-2]<para>______a = [Output Class Count (<see cref="MLDatas.OutputClassCount"/>)]<para>__________Range: <see cref="UpperThetas"/>[l-1]</para></para></para></para></para></para></para>
                /// <para>n -> Features <para>__'Features' Lenght:<para>______n = [Features Count (<see cref="MLDatas.FeaturesCount"/>) + unitary term (1)]</para></para></para>
                /// <para>*Warning: The last l index value is the output theta values index</para>
                /// </summary>
                public double[][][] UpperThetas { get; private set; }//Aqui contêm os valores de theta para cada unidade de ativação e o ultimo endereço possui a matriz que armazena os dados da camada de saída (Output)
                /// <summary>
                /// Activation unit array (double[l][m][n])
                /// <para>Where: </para>
                /// <para>l -> Layers <para>__'Layers' Lenght:<para>______l = [Input Layer (1) + Hidden Layers (<see cref="HiddenLayers"/>) + Output Layer (1)]</para></para></para>
                /// <para>m -> Elements <para>__'Elements' Lenght:<para>______m = [Elements Count (<see cref="MLDatas.ElementsCount"/>)]</para></para></para>
                /// <para>n -> Features <para>__'Features' Lenght:<para>______n = [Features Count (<see cref="MLDatas.FeaturesCount"/>) + bias unit (1)]<para>__________Range: <see cref="ActivationUnits"/>[0]<para>______n = [Hidden Layers Units Count (variable) + bias unit (1)]<para>__________Range: <see cref="ActivationUnits"/>[1] to <see cref="ActivationUnits"/>[l-2]<para>______n = [Output Class Count (<see cref="MLDatas.OutputClassCount"/>)]<para>__________Range: <see cref="ActivationUnits"/>[l-1]</para></para></para></para></para></para></para></para>
                /// </summary>
                public double[][][] ActivationUnits { get; private set; }//Aqui contêm o resultado para aplicação da estratégia backpropagation e o ultimo endereço possui a matriz que armazena os dados da camada de saída (Output)
                /// <summary>
                /// Error margins array (double[l][a])
                /// <para>Where: </para>
                /// <para>l -> Layers <para>__'Layers' Lenght:<para>______l = [Input Layer (1) + Hidden Layers (<see cref="HiddenLayers"/>) + Output Layer (1)]</para></para></para>
                /// <para>a -> Node (Actiovation Unit) <para>__'Node' Lenght:<para>______a = [Features Count (<see cref="MLDatas.FeaturesCount"/>) + bias unit (1)]<para>__________Range: <see cref="ActivationUnits"/>[0]<para>______a = [Hidden Layers Units Count (variable) + bias unit (1)]<para>__________Range: <see cref="ActivationUnits"/>[1] to <see cref="ActivationUnits"/>[l-2]<para>______a = [Output Class Count (<see cref="MLDatas.OutputClassCount"/>)]<para>__________Range: <see cref="ActivationUnits"/>[l-1]</para></para></para></para></para></para></para></para>
                /// </summary>
                public double[][] ErrorMargins { get; private set; }//Aqui contêm as margem de erro após o processo de fowardpropagation
                /// <summary>
                /// Derivations from backpropagation process (double[l][a][n])
                /// <para>Where: </para>
                /// <para>l -> Layers <para>__'Layers' Lenght:<para>______l = [Hidden Layers (<see cref="HiddenLayers"/>) + Output Layer (1)]</para></para></para>
                /// <para>a -> Node (Actiovation Unit) <para>__'Node' Lenght:<para>______a = [Features Count (<see cref="MLDatas.FeaturesCount"/>) + bias unit (1)] **for <see cref="int"/> hiddenLayers constructor argument value**<para>______a = [Hidden Layers Units Count (variable) + bias unit (1)] **for <see cref="int"/>[] hiddenLayers constructor argument value**<para>__________Range: <see cref="UpperThetas"/>[0] to <see cref="UpperThetas"/>[l-2]<para>______a = [Output Class Count (<see cref="MLDatas.OutputClassCount"/>)]<para>__________Range: <see cref="UpperThetas"/>[l-1]</para></para></para></para></para></para></para>
                /// <para>n -> Features <para>__'Features' Lenght:<para>______n = [Features Count (<see cref="MLDatas.FeaturesCount"/>) + unitary term (1)]</para></para></para>
                /// </summary>
                public double[][][] Derivatives { get; private set; }//Aqui contêm os valores das derivações parciais, produto do processo de backpropagation
                /// <summary>
                /// Delta values from backpropagation process  (double[l][a][n])
                /// <para>Where: </para>
                /// <para>l -> Layers <para>__'Layers' Lenght:<para>______l = [Hidden Layers (<see cref="HiddenLayers"/>) + Output Layer (1)]</para></para></para>
                /// <para>a -> Node (Actiovation Unit) <para>__'Node' Lenght:<para>______a = [Features Count (<see cref="MLDatas.FeaturesCount"/>) + bias unit (1)] **for <see cref="int"/> hiddenLayers constructor argument value**<para>______a = [Hidden Layers Units Count (variable) + bias unit (1)] **for <see cref="int"/>[] hiddenLayers constructor argument value**<para>__________Range: <see cref="UpperThetas"/>[0] to <see cref="UpperThetas"/>[l-2]<para>______a = [Output Class Count (<see cref="MLDatas.OutputClassCount"/>)]<para>__________Range: <see cref="UpperThetas"/>[l-1]</para></para></para></para></para></para></para>
                /// <para>n -> Features <para>__'Features' Lenght:<para>______n = [Features Count (<see cref="MLDatas.FeaturesCount"/>) + unitary term (1)]</para></para></para>
                /// </summary>
                public double[][][] Deltas { get; private set; }//Aqui contêm os valores deltas, produto do processo de backpropagation

                /// <summary>
                /// Total lenght of <see cref="UpperThetas"/> jagged array
                /// </summary>
                public double UpperThetasTotalLenght { get; private set; } = 0;
                /// <summary>
                /// Total lenght of <see cref="ActivationUnits"/> jagged array
                /// </summary>
                public double ActivationUnitsTotalLenght { get; private set; } = 0;
                /// <summary>
                /// Total lenght of <see cref="ErrorMargins"/> jagged array
                /// </summary>
                public double ErrorMarginsTotalLenght { get; private set; } = 0;
                /// <summary>
                /// Total lenght of <see cref="Derivatives"/> jagged array
                /// </summary>
                public double DerivativesTotalLenght { get; private set; } = 0;
                /// <summary>
                /// Total lenght of <see cref="Deltas"/> jagged array
                /// </summary>
                public double DeltasTotalLenght { get; private set; } = 0;

                /// <summary>
                /// This instance is allowed to use CUDA methods automatically, when is possible to use
                /// </summary>
                public bool CUDA { get; set; } = false;

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
                /// Elements count
                /// </summary>
                public int ElementsCount { get; set; }

                /// <summary>
                /// Output class count
                /// </summary>
                public int OutputClassCount { get; set; }

                /// <summary>
                /// Return true if this instance is using train elements, otherwise, development elements are being used.
                /// </summary>
                public bool IsTrainElements { get; set; }

                /// <summary>
                /// Hidden layers count
                /// </summary>
                public int HiddenLayers { get; private set; }

                /// <summary>
                /// Total layers count
                /// </summary>
                public int Layers { get; private set; }

                /// <summary>
                /// Units count per layer (withouth bias unit (+1))
                /// <para>[0] - Input layer units count</para>
                /// <para>[1] - 1º Hiden layer units count</para>
                /// <para>.<para>.<para>.</para></para></para>
                /// <para>[L - 2] - (L - 2)º Hiden layer units count</para>
                /// <para>[L - 1] - (L - 1)º Hiden layer units count</para>
                /// <para>[L] - Output layer units count</para>
                /// </summary>
                public int[] LayersUnitsCount { get; private set; }

                /// <summary>
                /// Activation function used on this instance
                /// </summary>
                public ActivationFunctions.ActiovationFunctions ActiovationFunction { get; set; }


                /// <summary>
                /// Creat a new neural networks instance
                /// </summary>
                public NeuralNetworks() { }
                /// <summary>
                /// Creat a new neural networks instance
                /// </summary>
                /// <param name="datas">Datas for train, test and development</param>
                /// <param name="alpha"></param>
                /// <param name="lambda"></param>
                /// <param name="hiddenLayers">Hidden layers quantity</param>
                /// <param name="cudaMethods">Creat and fill all arrays using <see cref="CUDACreatArrays(int[], int, int[], int, double[][])"/></param>
                /// <param name="useTrainElements">Use train elements, otherwise, will be used the delevopment elements</param>
                /// <param name="actiovationFunction">Activation function used on this instance</param>
                public NeuralNetworks(MLDatas datas, double alpha, double lambda, int hiddenLayers, bool cudaMethods, bool useTrainElements = true, ActivationFunctions.ActiovationFunctions actiovationFunction = ActivationFunctions.ActiovationFunctions.Sigmoid)
                {
                    Datas = datas.WiseClone();
                    CUDA = cudaMethods;

                    Datas.ProportionChanged += ResetConstructor;
                    Datas.SetsShuffled += ResetConstructor;


                    Alpha = alpha;
                    Lambda = lambda;
                    HiddenLayers = hiddenLayers;
                    Layers = HiddenLayers + 2;

                    LayersUnitsCount = new int[Layers];
                    LayersUnitsCount[0] = datas.FeaturesCount;
                    LayersUnitsCount[Layers - 1] = datas.OutputClassCount;

                    IsTrainElements = useTrainElements;

                    ActiovationFunction = actiovationFunction;

                    //*AVISO*
                    //A QUANTIDADE DE UNIDADES DE ATIVAÇÃO DENTRO DAS CAMADAS ESCONDIDAS SERÃO IGUAIS A QUANTIDADE INFORMADA NO PARÂMETRO 'hiddenLayers' + 1. ESSE '+ 1' REPRESENTA O ACRÉSCIMO DA UNIDADE DE POLARIZAÇÃO


                    //Aqui está sendo definido as dimensões para as matrizes que irão armazenar os valores de theta, tanto das cadamada escondidas, quanto da camada de saída
                    //1° Cria-se as matrizes para as camadas escondidas, com as seguintes dimensões:
                    //  hiddenLayers + 1 -> a matriz como um todo, irá permitir a inserção de matriz de theta das camadas escondidas e uma matriz que irá receber os valores na camada de saída
                    //  datas.FeaturesCount + 1 -> com relação a quantidade de elementos por camada, iremos colocar a quantidade de recursos por elemento e incrementando mais um "recurso", pois se tornará um núcleo unitário
                    //  datas.FeaturesCount + 1-> com relação aos recursos em cada linha de elementos, coloca-se a quantidade de recursos que um elemento possui + 1 termo unitária (valendo 1)
                    //
                    //2° Cria-se a matriz que vai representar a camada de saída
                    //  datas.Outputs.First().Length -> com relação a quantidade de elementos por camada, iremos colocar a quantidade de respostas que deveremos fornecer (1 saída, 2 saídas, etc)
                    //  datas.FeaturesCount + 1-> com relação aos recursos em cada linha de elementos, coloca-se a quantidade de recursos que um elemento possui + 1 termo unitária (valendo 1)

                    //O mesmo procedimento se vale para a propriedade ActivationUnits, com exceção da utilização do elementos de treino, tendo em vista que só serão necessário a quantidade de camadas escondidas + 1 e a quantidade de recursos

                    int
                        thetaLayers = HiddenLayers + 1,/*camadas escondidas + camada de saída*/
                        thetaElementsHL = datas.FeaturesCount + 1,/*recursos + unidade de polarização*/ //HL -> Hidden Layers
                        thetaElementsO = datas.OutputClassCount,/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs
                        thetaFeatures = datas.FeaturesCount + 1;/*recursos + termo unitário*/
                    int
                        derivativeLayers = thetaLayers,
                        derivativeElementsHL = thetaElementsHL,
                        derivativeElementsO = thetaElementsO,
                        derivativeFeatures = thetaFeatures;
                    int
                        deltaLayers = thetaLayers,
                        deltaElementsHL = thetaElementsHL,
                        deltaElementsO = thetaElementsO,
                        deltaFeatures = thetaFeatures;
                    int
                        errorMarginsLayers = thetaLayers,
                        errorMarginsFeaturesHL = datas.FeaturesCount + 1,/*recursos + termo unitário*/ //HL -> Hidden Layers
                        errorMarginsFeaturesO = datas.OutputClassCount;/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs
                    int
                        activationUnitsLayers = Layers,/*camada de entrada + camadas escondidas + camada de saída*/
                        activationUnitsFeaturesI = datas.FeaturesCount + 1,/*recursos + termo unitário*/ //I -> Input Layer
                        activationUnitsFeaturesHL = datas.FeaturesCount + 1,/*recursos + termo unitário*/ //HL -> Hidden Layers
                        activationUnitsFeaturesO = datas.OutputClassCount;/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs

                    #region Old Process
                    //if (cudaFill)
                    //{
                    //    UpperThetas = MachineLearningTools.GeneralTools.CUDACreatFillArray(thetaLayers, thetaElementsHL, thetaFeatures, true);
                    //    UpperThetas[thetaLayers - 1] = MachineLearningTools.GeneralTools.CUDACreatFillArray(thetaElementsO, thetaFeatures, true);

                    //    Derivatives = MachineLearningTools.GeneralTools.CUDACreatFillArray(derivativeLayers, derivativeElementsHL, derivativeFeatures, false);
                    //    Derivatives[derivativeLayers - 1] = MachineLearningTools.GeneralTools.CUDACreatFillArray(derivativeElementsO, derivativeFeatures, false);

                    //    Deltas = MachineLearningTools.GeneralTools.CUDACreatFillArray(deltaLayers, deltaElementsHL, deltaFeatures, false);
                    //    Deltas[deltaLayers - 1] = MachineLearningTools.GeneralTools.CUDACreatFillArray(deltaElementsO, deltaFeatures, false);

                    //    ErrorMargins = MachineLearningTools.GeneralTools.CUDACreatFillArray(errorMarginsLayers, errorMarginsFeaturesHL, false);
                    //    ErrorMargins[errorMarginsLayers - 1] = MachineLearningTools.GeneralTools.CUDACreatFillArray(errorMarginsFeaturesO, false);

                    //    ActivationUnits = MachineLearningTools.GeneralTools.CUDACreatFillArray(activationUnitsLayers, activationUnitsFeaturesHL, false);
                    //    //ActivationUnits[0] = MachineLearningTools.GeneralTools.CUDACreatFillArray(activationUnitsFeaturesI, false); //Pode ser desativado pois o tamanho de recursos é igual ao das camadas escondidas
                    //    ActivationUnits[activationUnitsLayers - 1] = MachineLearningTools.GeneralTools.CUDACreatFillArray(activationUnitsFeaturesO, false);

                    //    Parallel.For(0, activationUnitsLayers - 1, i => ActivationUnits[i][0] = 1);
                    //}
                    //else
                    //{
                    //    UpperThetas = MachineLearningTools.GeneralTools.CreatFillArray(thetaLayers, thetaElementsHL, thetaFeatures, true);
                    //    UpperThetas[thetaLayers - 1] = MachineLearningTools.GeneralTools.CreatFillArray(thetaElementsO, thetaFeatures, true);

                    //    Derivatives = MachineLearningTools.GeneralTools.CreatFillArray(derivativeLayers, derivativeElementsHL, derivativeFeatures, false);
                    //    Derivatives[derivativeLayers - 1] = MachineLearningTools.GeneralTools.CreatFillArray(derivativeElementsO, derivativeFeatures, false);

                    //    Deltas = MachineLearningTools.GeneralTools.CreatFillArray(deltaLayers, deltaElementsHL, deltaFeatures, false);
                    //    Deltas[deltaLayers - 1] = MachineLearningTools.GeneralTools.CreatFillArray(deltaElementsO, deltaFeatures, false);

                    //    ErrorMargins = MachineLearningTools.GeneralTools.CreatFillArray(errorMarginsLayers, errorMarginsFeaturesHL, false);
                    //    ErrorMargins[errorMarginsLayers - 1] = MachineLearningTools.GeneralTools.CreatFillArray(errorMarginsFeaturesO, false);

                    //    ActivationUnits = MachineLearningTools.GeneralTools.CreatFillArray(activationUnitsLayers, activationUnitsFeaturesHL, false);
                    //    //ActivationUnits[0] = MachineLearningTools.GeneralTools.CreatFillArray(activationUnitsFeaturesI, false); //Pode ser desativado pois o tamanho de recursos é igual ao das camadas escondidas
                    //    ActivationUnits[activationUnitsLayers - 1] = MachineLearningTools.GeneralTools.CreatFillArray(activationUnitsFeaturesO, false);

                    //    for (int i = 0; i < activationUnitsLayers - 1; i++)
                    //        ActivationUnits[i][0] = 1;                        
                    //}
                    #endregion

                    ////Processo de criação de matriz otimizado                    
                    if (cudaMethods)
                    {
                        int[] tL = new int[thetaLayers];
                        int[] aUL = new int[activationUnitsLayers];
                        Parallel.For(0, tL.Length, i =>
                        {
                            if (i < tL.Length - 1)
                            {
                                LayersUnitsCount[i + 1] = thetaElementsHL - 1;
                                tL[i] = thetaElementsHL - 1;
                            }
                            else
                                tL[i] = thetaElementsO;
                        });
                        Parallel.For(0, aUL.Length, i => aUL[i] = i == 0 ? activationUnitsFeaturesI : i < aUL.Length - 1 ? activationUnitsFeaturesHL : activationUnitsFeaturesO);
                        var arrays = CUDACreatArrays(tL, thetaFeatures, aUL, datas.TrainElementsCount, datas.TrainFeatures);
                        UpperThetas = arrays.upperThetas;
                        UpperThetasTotalLenght = arrays.upperThetasTotalLenght;
                        Derivatives = arrays.derivatives;
                        DerivativesTotalLenght = arrays.derivativesTotalLenght;
                        Deltas = arrays.deltas;
                        DeltasTotalLenght = arrays.deltasTotalLenght;

                        ErrorMargins = arrays.errorMargins;
                        ErrorMarginsTotalLenght = arrays.errorMarginsTotalLenght;
                        ActivationUnits = arrays.activationUnits;
                        ActivationUnitsTotalLenght = arrays.activationUnitsTotalLenght;

                        ElementsCount = datas.TrainElementsCount;
                        OutputClassCount = datas.OutputClassCount;
                    }
                    else
                    {
                        int[] tL = new int[thetaLayers];
                        int[] aUL = new int[activationUnitsLayers];
                        for (int i = 0, i_ = 0; i < tL.Length || i_ < aUL.Length; i++, i_++)
                        {
                            if (i < tL.Length - 1)
                            {
                                LayersUnitsCount[i + 1] = thetaElementsHL - 1;
                                tL[i] = thetaElementsHL - 1;
                            }
                            else if (i < tL.Length)
                                tL[i] = thetaElementsO;

                            if (i_ < aUL.Length)
                                aUL[i_] = i_ == 0 ? activationUnitsFeaturesI : i_ < aUL.Length - 1 ? activationUnitsFeaturesHL : activationUnitsFeaturesO;
                        }
                        var arrays = CreatArrays(tL, thetaFeatures, aUL, datas.TrainElementsCount, datas.TrainFeatures);
                        UpperThetas = arrays.upperThetas;
                        UpperThetasTotalLenght = arrays.upperThetasTotalLenght;
                        Derivatives = arrays.derivatives;
                        DerivativesTotalLenght = arrays.derivativesTotalLenght;
                        Deltas = arrays.deltas;
                        DeltasTotalLenght = arrays.deltasTotalLenght;

                        ErrorMargins = arrays.errorMargins;
                        ErrorMarginsTotalLenght = arrays.errorMarginsTotalLenght;
                        ActivationUnits = arrays.activationUnits;
                        ActivationUnitsTotalLenght = arrays.activationUnitsTotalLenght;

                        ElementsCount = datas.TrainElementsCount;
                        OutputClassCount = datas.OutputClassCount;
                    }
                }
                /// <summary>
                /// Creat a new neural networks instance
                /// </summary>
                /// <param name="datas">Datas for train, test and development</param>
                /// <param name="alpha"></param>
                /// <param name="lambda"></param>
                /// <param name="hiddenLayers">Hidden layers array quantities</param>
                /// <param name="cudaMethods">Creat and fill all arrays using <see cref="CUDACreatArrays(int[], int, int[], int, double[][])"/></param>
                /// <param name="useTrainElements">Use train elements, otherwise, will be used the delevopment elements</param>
                /// <param name="actiovationFunction">Activation function used on this instance</param>
                public NeuralNetworks(MLDatas datas, double alpha, double lambda, int[] hiddenLayers, bool cudaMethods, bool useTrainElements = true, ActivationFunctions.ActiovationFunctions actiovationFunction = ActivationFunctions.ActiovationFunctions.Sigmoid)
                {
                    Datas = datas.WiseClone();
                    CUDA = cudaMethods;

                    Datas.ProportionChanged += ResetConstructor;
                    Datas.SetsShuffled += ResetConstructor;


                    Alpha = alpha;
                    Lambda = lambda;
                    HiddenLayers = hiddenLayers.Length;
                    Layers = HiddenLayers + 2;

                    LayersUnitsCount = new int[Layers];
                    LayersUnitsCount[0] = datas.FeaturesCount;
                    LayersUnitsCount[Layers - 1] = datas.OutputClassCount;

                    IsTrainElements = useTrainElements;

                    ActiovationFunction = actiovationFunction;

                    //Aqui está sendo definido as dimensões para as matrizes que irão armazenar os valores de theta, tanto das cadamada escondidas, quanto da camada de saída
                    //1° Cria-se as matrizes para as camadas escondidas, com as seguintes dimensões:
                    //  hiddenLayers + 1 -> a matriz como um todo, irá permitir a inserção de matriz de theta das camadas escondidas e uma matriz que irá receber os valores na camada de saída
                    //  datas.FeaturesCount + 1 -> com relação a quantidade de elementos por camada, iremos colocar a quantidade de recursos por elemento e incrementando mais um "recurso", pois se tornará um núcleo unitário
                    //  datas.FeaturesCount + 1-> com relação aos recursos em cada linha de elementos, coloca-se a quantidade de recursos que um elemento possui + 1 termo unitária (valendo 1)
                    //
                    //2° Cria-se a matriz que vai representar a camada de saída
                    //  datas.Outputs.First().Length -> com relação a quantidade de elementos por camada, iremos colocar a quantidade de respostas que deveremos fornecer (1 saída, 2 saídas, etc)
                    //  datas.FeaturesCount + 1-> com relação aos recursos em cada linha de elementos, coloca-se a quantidade de recursos que um elemento possui + 1 termo unitária (valendo 1)

                    //O mesmo procedimento se vale para a propriedade ActivationUnits, com exceção da utilização do elementos de treino, tendo em vista que só serão necessário a quantidade de camadas escondidas + 1 e a quantidade de recursos


                    int
                        thetaElementsO = datas.OutputClassCount,/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs
                        thetaFeatures = datas.FeaturesCount + 1;/*recursos + termo unitário*/
                    int[]
                        thetaLayers = hiddenLayers.Append(thetaElementsO).ToArray();/*camadas escondidas + camada de saída*/

                    int
                        activationUnitsFeaturesI = datas.FeaturesCount, /*recursos*/ //I -> Input Layer
                        activationUnitsFeaturesO = datas.OutputClassCount;/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs
                    int[]
                        activationUnitsLayers = new int[] { activationUnitsFeaturesI }.Concat(hiddenLayers).Append(activationUnitsFeaturesO).ToArray();/*camada de entrada + camadas escondidas + camada de saída*/

                    //Adicionando termos unitários (theta), unidades unitária (activation unity) e atribuindo o tamanho de unidades por camada escondida
                    if (cudaMethods)
                        Parallel.For(0, thetaLayers.Length - 1, i =>
                        {
                            LayersUnitsCount[i + 1] = hiddenLayers[i];

                            //thetaLayers[i]++; 
                            //activationUnitsLayers[i]++;
                            //if (i == thetaLayers.Length - 2) 
                            //    activationUnitsLayers[i+1]++; 
                        });
                    else
                        for (int i = 0; i < thetaLayers.Length - 1; i++)
                        {
                            LayersUnitsCount[i + 1] = hiddenLayers[i];

                            //thetaLayers[i]++;
                            //activationUnitsLayers[i]++; 
                            //if (i == thetaLayers.Length - 2) 
                            //    activationUnitsLayers[i + 1]++;
                        }

                    //int
                    //    derivativeFeatures = thetaFeatures;
                    //int[]
                    //    derivativeLayers = thetaLayers;

                    //int
                    //    deltaFeatures = thetaFeatures;
                    //int[]
                    //    deltaLayers = thetaLayers;


                    //int[]
                    //    errorMarginsLayers = thetaLayers;



                    #region Old Process
                    //if (cudaFill)
                    //{
                    //    UpperThetas = MachineLearningTools.GeneralTools.CUDACreatFillArray(thetaLayers, thetaFeatures, true);
                    //    Derivatives = MachineLearningTools.GeneralTools.CUDACreatFillArray(derivativeLayers, derivativeFeatures, false);
                    //    Deltas = MachineLearningTools.GeneralTools.CUDACreatFillArray(deltaLayers, deltaFeatures, false);

                    //    ActivationUnits = MachineLearningTools.GeneralTools.CUDACreatFillArray(activationUnitsLayers, false);
                    //    ErrorMargins = MachineLearningTools.GeneralTools.CUDACreatFillArray(errorMarginsLayers, false);

                    //    Parallel.For(0, activationUnitsLayers.Length - 1, i => ActivationUnits[i][0] = 1);
                    //}
                    //else
                    //{
                    //    UpperThetas = MachineLearningTools.GeneralTools.CreatFillArray(thetaLayers, thetaFeatures, true);
                    //    Derivatives = MachineLearningTools.GeneralTools.CreatFillArray(derivativeLayers, derivativeFeatures, false);
                    //    Deltas = MachineLearningTools.GeneralTools.CreatFillArray(deltaLayers, deltaFeatures, false);

                    //    ActivationUnits = MachineLearningTools.GeneralTools.CreatFillArray(activationUnitsLayers, false);
                    //    ErrorMargins = MachineLearningTools.GeneralTools.CreatFillArray(errorMarginsLayers, false);

                    //    for (int i = 0; i < activationUnitsLayers.Length - 1; i++)
                    //        ActivationUnits[i][0] = 1;
                    //}
                    #endregion

                    ////Processo de criação de matriz otimizado
                    if (cudaMethods)
                    {
                        ElementsCount = useTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount;
                        OutputClassCount = datas.OutputClassCount;

                        var arrays = CUDACreatArrays(thetaLayers,
                            thetaFeatures,
                            activationUnitsLayers,
                            useTrainElements ? datas.TrainElementsCount : datas.DevelopmentElementsCount,
                            useTrainElements ? datas.TrainFeatures : datas.DevelopmentFeatures);

                        UpperThetas = arrays.upperThetas;
                        UpperThetasTotalLenght = arrays.upperThetasTotalLenght;
                        Derivatives = arrays.derivatives;
                        DerivativesTotalLenght = arrays.derivativesTotalLenght;
                        Deltas = arrays.deltas;
                        DeltasTotalLenght = arrays.deltasTotalLenght;

                        ErrorMargins = arrays.errorMargins;
                        ErrorMarginsTotalLenght = arrays.errorMarginsTotalLenght;
                        ActivationUnits = arrays.activationUnits;
                        ActivationUnitsTotalLenght = arrays.activationUnitsTotalLenght;


                    }
                    else
                    {
                        ElementsCount = useTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount;
                        OutputClassCount = datas.OutputClassCount;

                        var arrays = CreatArrays(thetaLayers,
                            thetaFeatures,
                            activationUnitsLayers,
                            useTrainElements ? datas.TrainElementsCount : datas.DevelopmentElementsCount,
                            useTrainElements ? datas.TrainFeatures : datas.DevelopmentFeatures);

                        UpperThetas = arrays.upperThetas;
                        UpperThetasTotalLenght = arrays.upperThetasTotalLenght;
                        Derivatives = arrays.derivatives;
                        DerivativesTotalLenght = arrays.derivativesTotalLenght;
                        Deltas = arrays.deltas;
                        DeltasTotalLenght = arrays.deltasTotalLenght;

                        ErrorMargins = arrays.errorMargins;
                        ErrorMarginsTotalLenght = arrays.errorMarginsTotalLenght;
                        ActivationUnits = arrays.activationUnits;
                        ActivationUnitsTotalLenght = arrays.activationUnitsTotalLenght;

                    }
                }


                /// <summary>
                /// Creat all arrays for NeuralNetworks class
                /// </summary>
                /// <param name="thetaLE">Array with theta layers and elements quantity<para>Layers = <paramref name="thetaLE"/>.Length</para><para>Elements = <paramref name="thetaLE"/>[i]</para></param>
                /// <param name="thetaF">Theta features quantity</param>
                /// <param name="activationLF">Array with activation units layers and features quantity<para>Layers = <paramref name="activationLF"/>.Length</para><para>Features = <paramref name="activationLF"/>[i]</para></param>
                /// <param name="elementsCount">Elements count</param>
                /// <param name="elementsFeatures">Elements features</param>
                /// <returns><see cref="Tuple{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10}"/>
                /// <para>T1 -> <see cref="UpperThetas"/> array</para>
                /// <para>T2 -> <see cref="UpperThetasTotalLenght"/> value</para>
                /// <para>T3 -> <see cref="Derivatives"/> array</para>
                /// <para>T4 -> <see cref="DerivativesTotalLenght"/> value</para>
                /// <para>T5 -> <see cref="Deltas"/> array</para>
                /// <para>T6 -> <see cref="DeltasTotalLenght"/> value</para>
                /// <para>T7 -> <see cref="ErrorMargins"/> array</para>
                /// <para>T8 -> <see cref="ErrorMarginsTotalLenght"/> value</para>
                /// <para>T9 -> <see cref="ActivationUnits"/> array</para>
                /// <para>T10 -> <see cref="ActivationUnitsTotalLenght"/> value</para>
                /// </returns>
                public Tuple<double[][][], double, double[][][], double, double[][][], double, double[][], double, double[][][], double> CreatArrays(int[] thetaLE, int thetaF, int[] activationLF, int elementsCount, double[][] elementsFeatures)
                {
                    Random rnd = new Random(DateTime.Now.Millisecond);
                    double[][][] upperThetas = new double[thetaLE.Length][][];
                    double upperThetasTotalLenght = 0;
                    double[][][] derivatives = new double[thetaLE.Length][][];
                    double derivativesTotalLenght = 0;
                    double[][][] deltas = new double[thetaLE.Length][][];
                    double deltasTotalLenght = 0;
                    double[][] errorMargins = new double[thetaLE.Length][];
                    double errorMarginsTotalLenght = 0;

                    double[][][] activationUnits = new double[activationLF.Length][][];
                    double activationUnitsTotalLenght = 0;

                    for (int i = 0, i_ = 0; i < thetaLE.Length || i_ < activationLF.Length; i++, i_++)
                    {
                        bool
                            a = i < thetaLE.Length,
                            b = i_ < activationLF.Length;
                        if (a)
                        {
                            upperThetas[i] = new double[thetaLE[i]][];
                            derivatives[i] = new double[thetaLE[i]][];
                            deltas[i] = new double[thetaLE[i]][];


                            errorMargins[i] = new double[thetaLE[i]];
                        }
                        if (b)
                            activationUnits[i_] = new double[elementsCount][];


                        for (int j = 0, j_ = 0; (a && j < thetaLE[i]) || (b && j_ < elementsCount); j++, j_++)
                        {
                            bool
                                c = a && j < thetaLE[i],
                                d = b && j_ < elementsCount;
                            if (a && c)
                            {
                                int uTLength = i - 1 == -1 ? thetaF : thetaLE[i - 1] + 1;
                                int dDLength = i - 1 == -1 ? thetaF /*- 1*/ : thetaLE[i - 1] + 1; //Adicionado devido ao processo de backpropagation
                                upperThetas[i][j] = new double[uTLength];

                                derivatives[i][j] = new double[dDLength];
                                deltas[i][j] = new double[dDLength];

                                errorMargins[i][j] = 0;
                                for (int k = 0; k < uTLength; k++)
                                {
                                    upperThetas[i][j][k] = rnd.NextDouble();

                                    if (k < dDLength)
                                    {
                                        derivatives[i][j][k] = 0;
                                        deltas[i][j][k] = 0;
                                    }
                                }
                            }
                            if (b && d)
                            {
                                activationUnits[i_][j_] = new double[i_ < activationLF.Length - 1 ? activationLF[i_] + 1 : activationLF[i_]]; // + 1 representa a unidade unitária (bias term) que vai se encontra no endereço 0
                                for (int k_ = 0; k_ < (i_ < activationLF.Length - 1 ? activationLF[i_] + 1 : activationLF[i_]); k_++)
                                    //Aqui será feito o seguinte:
                                    //*Temos i camadas
                                    //*Para a camada i_ = 0, iremos colocar todos os valores correspondentes a matriz double[][] elementsFeatures

                                    //activationUnits[i][j][k] = i < activationLF.Length - 1 && k == 0 ? 1 : 0;

                                    activationUnits[i_][j_][k_] = i_ < activationLF.Length - 1 && k_ == 0 ? 1 : i_ == 0 ? elementsFeatures[j_][k_ - 1] : 0;
                            }
                        }
                    }

                    upperThetasTotalLenght = upperThetas.Sum(l => l.Sum(m => m.Length));
                    derivativesTotalLenght = upperThetasTotalLenght;
                    deltasTotalLenght = upperThetasTotalLenght;
                    errorMarginsTotalLenght = errorMargins.Sum(l => l.Length);
                    activationUnitsTotalLenght = activationUnits.Sum(l => l.Sum(m => m.Length));

                    return new Tuple<double[][][], double, double[][][], double, double[][][], double, double[][], double, double[][][], double>
                        (upperThetas, upperThetasTotalLenght,
                        derivatives, derivativesTotalLenght,
                        deltas, deltasTotalLenght,
                        errorMargins, errorMarginsTotalLenght,
                        activationUnits, activationUnitsTotalLenght);
                }
                /// <summary>
                /// Creat all arrays for NeuralNetworks class
                /// </summary>
                /// <param name="thetaLE">Array with theta layers and elements quantity<para>Layers = <paramref name="thetaLE"/>.Length</para><para>Elements = <paramref name="thetaLE"/>[i]</para></param>
                /// <param name="thetaF">Theta features quantity</param>
                /// <param name="activationLF">Array with activation units layers and features quantity<para>Layers = <paramref name="activationLF"/>.Length</para><para>Features = <paramref name="activationLF"/>[i]</para></param>
                /// <param name="elementsCount">Elements count</param>
                /// <param name="elementsFeatures">Elements features</param>                
                /// <returns><see cref="Tuple{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10}"/>
                /// <para>T1 -> <see cref="UpperThetas"/> array</para>
                /// <para>T2 -> <see cref="UpperThetasTotalLenght"/> value</para>
                /// <para>T3 -> <see cref="Derivatives"/> array</para>
                /// <para>T4 -> <see cref="DerivativesTotalLenght"/> value</para>
                /// <para>T5 -> <see cref="Deltas"/> array</para>
                /// <para>T6 -> <see cref="DeltasTotalLenght"/> value</para>
                /// <para>T7 -> <see cref="ErrorMargins"/> array</para>
                /// <para>T8 -> <see cref="ErrorMarginsTotalLenght"/> value</para>
                /// <para>T9 -> <see cref="ActivationUnits"/> array</para>
                /// <para>T10 -> <see cref="ActivationUnitsTotalLenght"/> value</para>
                /// </returns>
                public Tuple<double[][][], double, double[][][], double, double[][][], double, double[][], double, double[][][], double> CUDACreatArrays(int[] thetaLE, int thetaF, int[] activationLF, int elementsCount, double[][] elementsFeatures)
                {
                    Random rnd = new Random(DateTime.Now.Millisecond);
                    double[][][] upperThetas = new double[thetaLE.Length][][];
                    double upperThetasTotalLenght = 0;
                    double[][][] derivatives = new double[thetaLE.Length][][];
                    double derivativesTotalLenght = 0;
                    double[][][] deltas = new double[thetaLE.Length][][];
                    double deltasTotalLenght = 0;
                    double[][] errorMargins = new double[thetaLE.Length][];
                    double errorMarginsTotalLenght = 0;
                    double[][][] activationUnits = new double[activationLF.Length][][];
                    double activationUnitsTotalLenght = 0;

                    Parallel.For(0, thetaLE.Length, i =>
                    {
                        upperThetas[i] = new double[thetaLE[i]][];
                        derivatives[i] = new double[thetaLE[i]][];
                        deltas[i] = new double[thetaLE[i]][];

                        errorMargins[i] = new double[thetaLE[i]];

                        Parallel.For(0, thetaLE[i], j =>
                        {
                            int uTLength = i - 1 == -1 ? thetaF : thetaLE[i - 1] + 1;
                            int dDLength = i - 1 == -1 ? thetaF /*- 1*/ : thetaLE[i - 1] + 1; //Adicionado devido ao processo de backpropagation

                            upperThetas[i][j] = new double[uTLength];

                            derivatives[i][j] = new double[dDLength];
                            deltas[i][j] = new double[dDLength];

                            //thetaArrayLikeLenghts[i] = 

                            errorMargins[i][j] = 0;
                            Parallel.For(0, uTLength, k =>
                            {
                                upperThetas[i][j][k] = rnd.NextDouble();

                                if (k < dDLength)
                                {
                                    derivatives[i][j][k] = 0;
                                    deltas[i][j][k] = 0;
                                }
                            });
                        });
                    });

                    Parallel.For(0, activationLF.Length, i =>
                    {
                        activationUnits[i] = new double[elementsCount][];
                        Parallel.For(0, elementsCount, j =>
                        {
                            activationUnits[i][j] = new double[i < activationLF.Length - 1 ? activationLF[i] + 1 : activationLF[i]];// + 1 representa a unidade unitária (bias term) que vai se encontra no endereço 0
                            Parallel.For(0, i < activationLF.Length - 1 ? activationLF[i] + 1 : activationLF[i], k => //
                            {
                                //Aqui será feito o seguinte:
                                //*Temos i camadas
                                //*Para a cadama i = 0, iremos colocar todos os valores correspondentes a matriz double[][] elementsFeatures

                                //activationUnits[i][j][k] = i < activationLF.Length - 1 && k == 0 ? 1 : 0;

                                activationUnits[i][j][k] = i < activationLF.Length - 1 && k == 0 ? 1 : i == 0 ? elementsFeatures[j][k - 1] : 0;
                            });
                        });
                    });

                    upperThetasTotalLenght = upperThetas.Sum(l => l.Sum(m => m.Length));
                    derivativesTotalLenght = upperThetasTotalLenght;
                    deltasTotalLenght = upperThetasTotalLenght;
                    errorMarginsTotalLenght = errorMargins.Sum(l => l.Length);
                    activationUnitsTotalLenght = activationUnits.Sum(l => l.Sum(m => m.Length));

                    return new Tuple<double[][][], double, double[][][], double, double[][][], double, double[][], double, double[][][], double>
                        (upperThetas, upperThetasTotalLenght,
                        derivatives, derivativesTotalLenght,
                        deltas, deltasTotalLenght,
                        errorMargins, errorMarginsTotalLenght,
                        activationUnits, activationUnitsTotalLenght);
                }


                /// <summary>
                /// Reset all default proprieties from this instance.
                /// <para>*This method have CUDA methods*</para>
                /// </summary>
                private void ResetConstructor()
                {
                    int
                        thetaElementsO = Datas.OutputClassCount,/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs
                        thetaFeatures = Datas.FeaturesCount + 1;/*recursos + termo unitário*/
                    int[]
                        thetaLayers = LayersUnitsCount.Skip(1).ToArray();/*camadas escondidas + camada de saída*/

                    int
                        activationUnitsFeaturesI = Datas.FeaturesCount, /*recursos*/ //I -> Input Layer
                        activationUnitsFeaturesO = Datas.OutputClassCount;/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs
                    int[]
                        activationUnitsLayers = LayersUnitsCount;/*camada de entrada + camadas escondidas + camada de saída*/

                    ////Processo de criação de matriz otimizado
                    if (CUDA)
                    {
                        ElementsCount = IsTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount;
                        OutputClassCount = Datas.OutputClassCount;

                        var arrays = CUDACreatArrays(thetaLayers,
                            thetaFeatures,
                            activationUnitsLayers,
                            IsTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount,
                            IsTrainElements ? Datas.TrainFeatures : Datas.DevelopmentFeatures);

                        UpperThetas = arrays.upperThetas;
                        UpperThetasTotalLenght = arrays.upperThetasTotalLenght;
                        Derivatives = arrays.derivatives;
                        DerivativesTotalLenght = arrays.derivativesTotalLenght;
                        Deltas = arrays.deltas;
                        DeltasTotalLenght = arrays.deltasTotalLenght;

                        ErrorMargins = arrays.errorMargins;
                        ErrorMarginsTotalLenght = arrays.errorMarginsTotalLenght;
                        ActivationUnits = arrays.activationUnits;
                        ActivationUnitsTotalLenght = arrays.activationUnitsTotalLenght;

                    }
                    else
                    {
                        ElementsCount = IsTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount;
                        OutputClassCount = Datas.OutputClassCount;

                        var arrays = CreatArrays(thetaLayers,
                            thetaFeatures,
                            activationUnitsLayers,
                            IsTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount,
                            IsTrainElements ? Datas.TrainFeatures : Datas.DevelopmentFeatures);

                        UpperThetas = arrays.upperThetas;
                        UpperThetasTotalLenght = arrays.upperThetasTotalLenght;
                        Derivatives = arrays.derivatives;
                        DerivativesTotalLenght = arrays.derivativesTotalLenght;
                        Deltas = arrays.deltas;
                        DeltasTotalLenght = arrays.deltasTotalLenght;

                        ErrorMargins = arrays.errorMargins;
                        ErrorMarginsTotalLenght = arrays.errorMarginsTotalLenght;
                        ActivationUnits = arrays.activationUnits;
                        ActivationUnitsTotalLenght = arrays.activationUnitsTotalLenght;


                    }
                }
                /// <summary>
                /// Reset all default proprieties from this instance
                /// <para>*This method have CUDA methods*</para>
                /// </summary>
                /// <param name="useTrainElements">Use train elements, otherwise, will be used the delevopment elements</param>
                private void ResetConstructor(bool useTrainElements)
                {

                    IsTrainElements = useTrainElements;
                    int
                        thetaElementsO = Datas.OutputClassCount,/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs
                        thetaFeatures = Datas.FeaturesCount + 1;/*recursos + termo unitário*/
                    int[]
                        thetaLayers = LayersUnitsCount.Skip(1).ToArray();/*camadas escondidas + camada de saída*/

                    int
                        activationUnitsFeaturesI = Datas.FeaturesCount, /*recursos*/ //I -> Input Layer
                        activationUnitsFeaturesO = Datas.OutputClassCount;/*quantidade de respostas finais(quantidade de classes de classificação)*/ //O -> Outputs
                    int[]
                        activationUnitsLayers = LayersUnitsCount;/*camada de entrada + camadas escondidas + camada de saída*/

                    ////Processo de criação de matriz otimizado
                    if (CUDA)
                    {
                        ElementsCount = IsTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount;
                        OutputClassCount = Datas.OutputClassCount;

                        var arrays = CUDACreatArrays(thetaLayers,
                            thetaFeatures,
                            activationUnitsLayers,
                            IsTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount,
                            IsTrainElements ? Datas.TrainFeatures : Datas.DevelopmentFeatures);

                        UpperThetas = arrays.upperThetas;
                        UpperThetasTotalLenght = arrays.upperThetasTotalLenght;
                        Derivatives = arrays.derivatives;
                        DerivativesTotalLenght = arrays.derivativesTotalLenght;
                        Deltas = arrays.deltas;
                        DeltasTotalLenght = arrays.deltasTotalLenght;

                        ErrorMargins = arrays.errorMargins;
                        ErrorMarginsTotalLenght = arrays.errorMarginsTotalLenght;
                        ActivationUnits = arrays.activationUnits;
                        ActivationUnitsTotalLenght = arrays.activationUnitsTotalLenght;

                    }
                    else
                    {
                        ElementsCount = IsTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount;
                        OutputClassCount = Datas.OutputClassCount;

                        var arrays = CreatArrays(thetaLayers,
                            thetaFeatures,
                            activationUnitsLayers,
                            IsTrainElements ? Datas.TrainElementsCount : Datas.DevelopmentElementsCount,
                            IsTrainElements ? Datas.TrainFeatures : Datas.DevelopmentFeatures);

                        UpperThetas = arrays.upperThetas;
                        UpperThetasTotalLenght = arrays.upperThetasTotalLenght;
                        Derivatives = arrays.derivatives;
                        DerivativesTotalLenght = arrays.derivativesTotalLenght;
                        Deltas = arrays.deltas;
                        DeltasTotalLenght = arrays.deltasTotalLenght;

                        ErrorMargins = arrays.errorMargins;
                        ErrorMarginsTotalLenght = arrays.errorMarginsTotalLenght;
                        ActivationUnits = arrays.activationUnits;
                        ActivationUnitsTotalLenght = arrays.activationUnitsTotalLenght;


                    }
                }


                /// <summary>
                /// Do foward propagation,backward propagation and gradient descent using all elements per propagation (Epoch). 
                /// <para>*This method have CUDA methods*</para>
                /// </summary>
                /// <param name="propagationsCount">How many times will this neural network execute the following methods respectively:
                /// <para><see cref="DoFowardPropagation()"/></para>
                /// <para><see cref="DoBackwardPropagation()"/></para>
                /// <para><see cref="DoGradientDescent()"/></para></param>
                /// <param name="resetThetas">Set new random values on <see cref="UpperThetas"/> propiety</param>
                public void DoPropagation(int propagationsCount, bool resetThetas = true)
                {
                    if (resetThetas && UpperThetas != null)
                    {
                        double[][][] upperThetas = UpperThetas;
                        if (CUDA)
                            MachineLearningTools.GeneralTools.CUDAFillArray(ref upperThetas, true);
                        else
                            MachineLearningTools.GeneralTools.FillArray(ref upperThetas, true);
                    }

                    for (int prop = 0; prop < propagationsCount; prop++)
                    {
                        DoFowardPropagation();
                        DoBackwardPropagation();
                        if (CUDA)
                            CUDADoGradientDescent();
                        else
                            DoGradientDescent();
                    }
                }
                /// <summary>
                /// Do foward propagation,backward propagation and gradient descent per batch for all elements per propagation (Batch). 
                /// <para>*This method have CUDA methods*</para>
                /// </summary>
                /// <param name="propagationsCount">How many times will this neural network execute the following methods respectively:
                /// <para><see cref="DoFowardPropagation()"/></para>
                /// <para><see cref="DoBackwardPropagation()"/></para>
                /// <para><see cref="DoGradientDescent()"/></para></param>
                /// <param name="batchCount">Batch count</param>
                /// <param name="resetThetas">Set new random values on <see cref="UpperThetas"/> propiety</param>
                public void DoPropagation(int propagationsCount, int batchCount, bool resetThetas = true)
                {
                    if (resetThetas && UpperThetas != null)
                    {
                        double[][][] upperThetas = UpperThetas;
                        if (CUDA)
                            MachineLearningTools.GeneralTools.CUDAFillArray(ref upperThetas, true);
                        else
                            MachineLearningTools.GeneralTools.FillArray(ref upperThetas, true);
                    }

                    for (int prop = 0; prop < propagationsCount; prop++)
                        for (int batch = 0, bStack = batchCount; batch < (bStack > ElementsCount ? ElementsCount : bStack); batch += batchCount, bStack += batchCount)
                        {
                            DoFowardPropagation(batch/*Initial Index*/, (bStack > ElementsCount ? ElementsCount : bStack)/*Count*/);
                            DoBackwardPropagation(batch/*Initial Index*/, batchCount/*Count*/);
                            if (CUDA)
                                CUDADoGradientDescent();
                            else
                                DoGradientDescent();
                        }
                }


                /// <summary>
                /// Set Actiovation Units propriety jagged array
                /// </summary>
                public void DoFowardPropagation()
                {
                    Func<double, double> aF = ActivationFunctions.GetFunction(ActiovationFunction);//Retorna uma função de ativação (Sigmoid, tangente hiperbólica, retificação linear, etc)                   
                    //Como vai funcionar aqui:
                    //1º 'for': Ele vai permitir passar por cada camada, com exceção da camada de saída, das unidades de ativação
                    //2º 'for': Ele vai permitir passar por cada conjunto de valores Theta, preparado para cada unidade de ativação em cada camada. Então se, por exemplo, tivermos na camada l=1, 10 unidades de ativação, então no enedereço UpperTheta[0] teremos 10 matrizes, 1 para cada unidade de ativação
                    //3º 'for': Ele vai permitir passar por cada elemento (conjunto de valores X) á partir do tamanho da matriz de ativação na camada l
                    //.
                    //Após ativado as 3 estruturas de repetição, iremos adicionar a matriz de unidades de ativação posterior, os valores, resultado da função sigmoid


                    /*VISUALIZAÇÃO*///List<string> lines = new List<string>();
                    /*VISUALIZAÇÃO*///string[][][] elements = ActivationUnits.ToList().ConvertAll(x => x.ToList().ConvertAll(y => y.ToList().ConvertAll(z => z.ToString()).ToArray()).ToArray()).ToArray();

                    for (int l = 0; l < ActivationUnits.Length - 1; l++)//Passando por cada camada das unidades de ativação antes da última camada, a camada de saída
                    {
                        /*VISUALIZAÇÃO*///Console.WriteLine($"Na camada {l+1}, para cada uma das {UpperThetas[l].Length} unidades de ativação, serão utilizados {ActivationUnits[l].Length} elementos, onde cada elemento possui {UpperThetas[l][0].Length} recursos, sendo 1 unitário (valendo 1)");

                        /*VISUALIZAÇÃO*///Console.WriteLine($"Camada: {l+1}");
                        for (int a = 0; a < UpperThetas[l].Length; a++)//Passando por cada matriz theta, correspondente a cada unidade de ativação da camada l
                        {
                            /*VISUALIZAÇÃO*///Console.WriteLine($"    Unidade de ativação: {a}");
                            for (int m = 0; m < ActivationUnits[l].Length; m++)//Criando as predições onde cada endereço z dentro da matriz predictions é a predição (thetas, x's) 
                            {
                                /*VISUALIZAÇÃO*///Console.WriteLine($"        Elemento: {m}");
                                /*VISUALIZAÇÃO*///Console.WriteLine(                                            /*$"            Camada: {l}   Recurso: {n}    Elemento: {m}\n" +*/                                            $"          Valores theta para unidade ativação ({a}): {string.Join("    ", UpperThetas[l][a].ToList().ConvertAll(uT => Math.Round(uT,2)))}\n\n" +                                            $"          valores armazenados na unidade de ativação da camada {l} endereço {m} (excluindo valor unitário): {string.Join("    ", ActivationUnits[l][m].ToList().ConvertAll(aU => Math.Round(aU, 2)))}\n\n" +                                            $"          Prediction = {Predict(UpperThetas[l][a], ActivationUnits[l][m])}\n" +                                            $"          Sigmoid = {Sigmoid(Predict(UpperThetas[l][a], ActivationUnits[l][m]))}\n\n"                                        );
                                /*VISUALIZAÇÃO*///lines.Add($"a[{l + 1}][{m}][{a}] = a[{l}][{m}] X Θ[{l}][{a}]");
                                /*VISUALIZAÇÃO*///elements[l + 1][m][a] = $"a[{l}][{m}] X Θ[{l}][{a}]";

                                ActivationUnits[l + 1][m][a] = aF(HypothesisAU(UpperThetas[l][a], ActivationUnits[l][m]));
                            }
                        }
                    }

                    /*VISUALIZAÇÃO*///System.Windows.Forms.Clipboard.SetText(string.Join("\n", lines));
                    /*VISUALIZAÇÃO*///Console.WriteLine("\n\n\n\n\n\n\n");
                    /*VISUALIZAÇÃO*///Console.WriteLine("\n\nActivation Units");
                    /*VISUALIZAÇÃO*///Console.WriteLine(string.Join("\n", ActivationUnits.ToList().ConvertAll(i => string.Join("\n", i.ToList().ConvertAll(j => string.Join(";", j.ToList().ConvertAll(k => Math.Round(k, 2))))))));
                    /*VISUALIZAÇÃO*///Console.WriteLine("\n\n\nActivation Units");
                    /*VISUALIZAÇÃO*///Console.WriteLine(string.Join("\n", elements.ToList().ConvertAll(i => string.Join("\n", i.ToList().ConvertAll(j => string.Join(";", j))))));
                    /*VISUALIZAÇÃO*///Console.WriteLine("\n\n\n\n\n\n\n");

                }
                /// <summary>
                /// Set Actiovation Units propriety jagged array
                /// </summary>
                /// <param name="initialIndex">Elements initial index</param>
                /// <param name="count">Elements count</param>
                public void DoFowardPropagation(int initialIndex, int count)
                {
                    Func<double, double> aF = ActivationFunctions.GetFunction(ActiovationFunction);//Retorna uma função de ativação (Sigmoid, tangente hiperbólica, retificação linear, etc)                   
                    for (int l = 0; l < ActivationUnits.Length - 1; l++)//Passando por cada camada das unidades de ativação antes da última camada, a camada de saída
                        for (int a = 0; a < UpperThetas[l].Length; a++)//Passando por cada matriz theta, correspondente a cada unidade de ativação da camada l
                            for (int m = initialIndex; m < initialIndex + count; m++)//Criando as predições onde cada endereço z dentro da matriz predictions é a predição (thetas, x's) 
                                ActivationUnits[l + 1][m][a] = aF(HypothesisAU(UpperThetas[l][a], ActivationUnits[l][m]));
                }
                /// <summary>
                /// Set Actiovation Units propriety jagged array
                /// </summary>
                /// <param name="activationUnits">Activation units jagged array</param>
                /// <param name="upperThetas">Upper thetas jagged array</param>
                public void DoFowardPropagation(ref double[][][] activationUnits, double[][][] upperThetas)
                {
                    Func<double, double> aF = ActivationFunctions.GetFunction(ActiovationFunction);//Retorna uma função de ativação (Sigmoid, tangente hiperbólica, retificação linear, etc)                   
                    for (int l = 0; l < activationUnits.Length - 1; l++)//Passando por cada camada das unidades de ativação antes da última camada, a camada de saída
                        for (int a = 0; a < upperThetas[l].Length; a++)//Passando por cada matriz theta, correspondente a cada unidade de ativação da camada l
                            for (int m = 0; m < activationUnits[l].Length; m++)//Criando as predições onde cada endereço z dentro da matriz predictions é a predição (thetas, x's) 
                                activationUnits[l + 1][m][a] = aF(HypothesisAU(upperThetas[l][a], activationUnits[l][m]));
                }


                /// <summary>
                /// Set error margins, deltas and derivatives propriety jagged array and updat upper thetas propriety values
                /// </summary>
                public void DoBackwardPropagation()
                {
                    //Aqui será feito a atribuição dos valore de:
                    //Margem de erro
                    //Derivações
                    //Deltas
                    //Thetas

                    //1° Etapa: Definição dos valore de margem de erro
                    //  Existem duas formulas que serão consideras, a 1° formula, que iniciará na camada de saída e a 2° formula, que será aplicada para outras camadas
                    //  1° Formula
                    //  δ[l][n] = a[l][n] - y[n]
                    //      Onde:
                    //          δ -> margem de erro
                    //          a -> unidades de ativação
                    //          y -> saída (ou 'resposta esperada')
                    //          l -> camada (abrange desde a camada 1 até a última camada)
                    //          n -> para o caso de 'δ' e 'a', seria o endereço da unidade de ativação (ou 'node'), para o caso de 'y', seria a classe da saída (exemplo.: identificar se é cão (1° classe), gato (2° classe) ou pato  (3° classe), portanto, y possui um valor 'n' que vai de 0 á 2)
                    //  2° Formula
                    //  δ[l][n] = (Θ[l])ᵗ * δ[l+1] .* g'(Z[l]) 
                    //      g'(Z[l]) =  a[l] .* (1 - a[l])
                    //      Onde:
                    //          (X)ᵗ -> Matriz transposta de X *Irrelevante para como vou fazer usando C#, somente para compreenssão*
                    //          Θ -> pesos theta
                    //          g'(Z[l]) -> função derivada, que é o produto      

                    double[][] outputs = Datas.TrainOutputs;
                    for (int m = 0; m < ElementsCount; m++)
                    {
                        /*VISUALIZAÇÃO*///Console.WriteLine("\n\n\n\n\n\n");

                        for (int l = ErrorMargins.Length - 1; l > -1; l--)
                        {
                            if (l == ErrorMargins.Length - 1)
                                for (int n = 0; n < ErrorMargins[l].Length; n++)
                                    ErrorMargins[l][n] = ActivationUnits[l][m][n] - outputs[m][n];
                            else
                            {
                                double[] uTxEM = MachineLearningTools.GeneralTools.Multiply(MachineLearningTools.GeneralTools.Transpose(UpperThetas[l + 1]), ErrorMargins[l + 1]/*, 1*/);
                                for (int n = 0; n < ErrorMargins[l].Length; n++)
                                    ErrorMargins[l][n] = uTxEM[n] * (ActivationUnits[l + 1][m][n] * (1 - ActivationUnits[l + 1][m][n]));
                            }
                        }

                        /*VISUALIZAÇÃO*///Console.WriteLine("\n\nError Margins");
                        /*VISUALIZAÇÃO*///Console.WriteLine(string.Join("\n", ErrorMargins.ToList().ConvertAll(i => string.Join("    ", i))));
                        /*VISUALIZAÇÃO*///Console.ReadKey();

                        for (int l = 0; l < Deltas.Length; l++)
                            for (int a = 0; a < Deltas[l].Length; a++)
                                for (int n = 0; n < Deltas[l][a].Length; n++)
                                    Deltas[l][a][n] = Deltas[l][a][n] + ActivationUnits[l][m][n] * ErrorMargins[l][a];

                        /*VISUALIZAÇÃO*///Console.WriteLine("\n\nDeltas");
                        /*VISUALIZAÇÃO*///Console.WriteLine(string.Join("\n\n", Deltas.ToList().ConvertAll(i => string.Join("\n", i.ToList().ConvertAll(j => string.Join("    ", j.ToList().ConvertAll(k => Math.Round(k, 2))))))));
                        /*VISUALIZAÇÃO*///Console.ReadKey();

                        for (int l = 0; l < Derivatives.Length; l++)
                            for (int a = 0; a < Derivatives[l].Length; a++)
                                for (int n = 0; n < Derivatives[l][a].Length; n++)
                                    Derivatives[l][a][n] = n == 0 ? (1 / ElementsCount) * Deltas[l][a][n] : (1 / ElementsCount) * Deltas[l][a][n] + Lambda * UpperThetas[l][a][n];

                        /*VISUALIZAÇÃO*///Console.WriteLine("\n\nDerivatives");
                        /*VISUALIZAÇÃO*///Console.WriteLine(string.Join("\n\n", Derivatives.ToList().ConvertAll(i => string.Join("\n", i.ToList().ConvertAll(j => string.Join("   ", j.ToList().ConvertAll(k => Math.Round(k, 2))))))));
                        /*VISUALIZAÇÃO*///Console.ReadKey();
                    }
                }
                /// <summary>
                /// Set error margins, deltas and derivatives propriety jagged array and updat upper thetas propriety values
                /// </summary>
                /// <param name="initialIndex">Elements initial index</param>
                /// <param name="count">Elements count</param>
                public void DoBackwardPropagation(int initialIndex, int count)
                {

                    double[][] outputs = Datas.TrainOutputs;
                    for (int m = initialIndex; m < initialIndex + count; m++)
                    {
                        for (int l = ErrorMargins.Length - 1; l > -1; l--)
                        {
                            if (l == ErrorMargins.Length - 1)
                                for (int n = 0; n < ErrorMargins[l].Length; n++)
                                    ErrorMargins[l][n] = ActivationUnits[l][m][n] - outputs[m][n];
                            else
                            {
                                double[] uTxEM = MachineLearningTools.GeneralTools.Multiply(MachineLearningTools.GeneralTools.Transpose(UpperThetas[l + 1]), ErrorMargins[l + 1]/*, 1*/);
                                for (int n = 0; n < ErrorMargins[l].Length; n++)
                                    ErrorMargins[l][n] = uTxEM[n] * (ActivationUnits[l + 1][m][n] * (1 - ActivationUnits[l + 1][m][n]));
                            }
                        }
                        for (int l = 0; l < Deltas.Length; l++)
                            for (int a = 0; a < Deltas[l].Length; a++)
                                for (int n = 0; n < Deltas[l][a].Length; n++)
                                    Deltas[l][a][n] = Deltas[l][a][n] + ActivationUnits[l][m][n] * ErrorMargins[l][a];
                        for (int l = 0; l < Derivatives.Length; l++)
                            for (int a = 0; a < Derivatives[l].Length; a++)
                                for (int n = 0; n < Derivatives[l][a].Length; n++)
                                    Derivatives[l][a][n] = n == 0 ? (1 / ElementsCount) * Deltas[l][a][n] : (1 / ElementsCount) * Deltas[l][a][n] + Lambda * UpperThetas[l][a][n];
                    }
                }
                /// <summary>
                /// Set error margins, deltas and derivatives propriety jagged array and updat upper thetas propriety values
                /// </summary>
                /// <param name="upperThetas"></param>
                /// <param name="activationUnits"></param>
                /// <param name="outputs"></param>
                /// <param name="deltas"></param>
                /// <param name="derivatives"></param>
                /// <param name="errorMargins"></param>
                /// <param name="lambda"></param>
                public void DoBackwardPropagation(double[][][] upperThetas, double[][][] activationUnits, double[][] outputs, double[][][] deltas, double[][][] derivatives, double[][] errorMargins, double lambda)
                {

                    for (int m = 0; m < outputs.Length; m++)
                    {

                        for (int l = errorMargins.Length - 1; l > -1; l--)
                        {
                            if (l == errorMargins.Length - 1)
                                for (int n = 0; n < errorMargins[l].Length; n++)
                                    errorMargins[l][n] = activationUnits[l][m][n] - outputs[m][n];
                            else
                            {
                                double[] uTxEM = MachineLearningTools.GeneralTools.Multiply(MachineLearningTools.GeneralTools.Transpose(upperThetas[l + 1]), errorMargins[l + 1]/*, 1*/);
                                for (int n = 0; n < errorMargins[l].Length; n++)
                                    errorMargins[l][n] = uTxEM[n] * (activationUnits[l + 1][m][n] * (1 - activationUnits[l + 1][m][n]));
                            }
                        }

                        for (int l = 0; l < deltas.Length; l++)
                            for (int a = 0; a < deltas[l].Length; a++)
                                for (int n = 0; n < deltas[l][a].Length; n++)
                                    deltas[l][a][n] = deltas[l][a][n] + activationUnits[l][m][n] * errorMargins[l][a];

                        for (int l = 0; l < derivatives.Length; l++)
                            for (int a = 0; a < derivatives[l].Length; a++)
                                for (int n = 0; n < derivatives[l][a].Length; n++)
                                    derivatives[l][a][n] = n == 0 ? (1 / outputs.Length) * deltas[l][a][n] : (1 / outputs.Length) * deltas[l][a][n] + lambda * upperThetas[l][a][n];
                    }
                }


                /// <summary>
                /// Gradient descent process to refresh <see cref="UpperThetas"/> values.
                /// <para>*Warning: This process will reset the <see cref="Derivatives"/> propriety values</para>
                /// </summary>
                public void DoGradientDescent()
                {
                    for (int l = 0; l < Derivatives.Length; l++)
                        for (int a = 0; a < Derivatives[l].Length; a++)
                            for (int n = 0; n < Derivatives[l][a].Length; n++)
                            {
                                UpperThetas[l][a][n] = UpperThetas[l][a][n] - Alpha * Derivatives[l][a][n];
                                Derivatives[l][a][n] = 0;
                            }
                }
                /// <summary>
                /// Gradient descent process to refresh <see cref="UpperThetas"/> values.
                /// <para>*Warning: This process will reset the <see cref="Derivatives"/> propriety values</para>
                /// </summary>
                public void CUDADoGradientDescent()
                {
                    Parallel.For(0, Derivatives.Length, l =>
                    {
                        Parallel.For(0, Derivatives[l].Length, a =>
                        {
                            Parallel.For(0, Derivatives[l][a].Length, n =>
                            {
                                UpperThetas[l][a][n] = UpperThetas[l][a][n] - Alpha * Derivatives[l][a][n];
                                Derivatives[l][a][n] = 0;
                            });
                        });
                    });
                }
                /// <summary>
                /// Gradient descent process to refresh <see cref="UpperThetas"/> values.
                /// <para>*Warning: This process will reset the <see cref="Derivatives"/> propriety values</para>
                /// </summary>
                public void DoGradientDescent(double[][][] upperThetas, double[][][] derivatives, double alpha)
                {
                    for (int l = 0; l < derivatives.Length; l++)
                        for (int a = 0; a < derivatives[l].Length; a++)
                            for (int n = 0; n < derivatives[l][a].Length; n++)
                            {
                                upperThetas[l][a][n] = upperThetas[l][a][n] - alpha * derivatives[l][a][n];
                                derivatives[l][a][n] = 0;
                            }
                }
                /// <summary>
                /// Gradient descent process to refresh <see cref="UpperThetas"/> values.
                /// <para>*Warning: This process will reset the <see cref="Derivatives"/> propriety values</para>
                /// </summary>
                public void CUDADoGradientDescent(double[][][] upperThetas, double[][][] derivatives, double alpha)
                {
                    Parallel.For(0, derivatives.Length, l =>
                    {
                        Parallel.For(0, derivatives[l].Length, a =>
                        {
                            Parallel.For(0, derivatives[l][a].Length, n =>
                            {
                                upperThetas[l][a][n] = upperThetas[l][a][n] - alpha * derivatives[l][a][n];
                                derivatives[l][a][n] = 0;
                            });
                        });
                    });
                }


                /// <summary>
                /// Return the cost function value from this instance. Using the currently <see cref="UpperThetas"/> array values, <see cref="MLDatas.TrainFeatures"/> and <see cref="MLDatas.TrainOutputs"/> array values from <see cref="Datas"/> propriety.
                /// </summary>
                /// <returns></returns>
                public double CostFunction()
                {
                    double sumCost()
                    {
                        double sum = 0;

                        double[][] outputs = Datas.TrainOutputs;
                        double[][] aUOutputs = ActivationUnits.Last();
                        for (int m = 0; m < ElementsCount; m++)
                            for (int k = 0; k < OutputClassCount; k++)
                                sum += outputs[m][k] * (-Math.Log(aUOutputs[m][k])) + (1 - outputs[m][k]) * (-Math.Log(1 - aUOutputs[m][k]));

                        return sum;
                    }

                    double sumThetaSquare() => UpperThetas.Sum(l => l.Sum(sl => sl.Skip(1)/*pula o valor theta que multplica o termo unitário*/.Sum(slp1 => Math.Pow(slp1, 2)))); /*{ double sum = 0; for (int l = 0; l < UpperThetas.Length; l++) for (int sl = 0; sl < UpperThetas[l].Length; sl++) for (int slp1 = 0; slp1 < UpperThetas[l][sl].Length; slp1++) sum += Math.Pow(UpperThetas[l][sl][slp1], 2); return sum; }*/

                    //SetActiovationUnits(); //Esse método define os valore nas unidades de ativação

                    double m_ = Datas.TrainElementsCount;
                    double sumC = sumCost();
                    double sumTS = sumThetaSquare();

                    #region Test
                    //Console.WriteLine();
                    //Console.WriteLine();
                    //Console.WriteLine($"m_ = {m_}");
                    //Console.WriteLine($"1/m_ = {1/m_}");
                    //Console.WriteLine($"sumCost() = {sumC}");
                    //Console.WriteLine($"Lambda = {Lambda}");
                    //Console.WriteLine($"1/(2*m_) = {1/(2*m_)}");
                    //Console.WriteLine($"sumThetaSquare() = {sumTS}");
                    //Console.WriteLine($"J = {(1 / m_) * sumC + (Lambda / (2 * m_)) * sumTS}");
                    //Console.WriteLine();
                    //Console.ReadLine();
                    #endregion

                    return (1 / m_) * sumC + (Lambda / (2 * m_)) * sumTS;
                }
                /// <summary>
                /// Return the cost function value.
                /// </summary>
                /// <param name="activationUnits">Activation units jagged array</param>
                /// <param name="upperThetas">Upper thetas jagged array</param>
                /// <param name="outputs">Outputs array</param>
                /// <returns></returns>
                public double CostFunction(double[][][] activationUnits, double[][][] upperThetas, double[][] outputs)
                {
                    double sumCost()
                    {
                        double sum = 0;
                        double[][] aUOutputs = activationUnits.Last();
                        for (int m = 0; m < outputs.Length; m++)
                            for (int k = 0; k < outputs[m].Length; k++)
                                sum += outputs[m][k] * (-Math.Log(aUOutputs[m][k])) + (1 - outputs[m][k]) * (-Math.Log(1 - aUOutputs[m][k]));
                        return sum;
                    }

                    double sumThetaSquare() => UpperThetas.Sum(l => l.Sum(sl => sl.Skip(1)/*pula o valor theta que multplica o termo unitário*/.Sum(slp1 => Math.Pow(slp1, 2)))); /*{ double sum = 0; for (int l = 0; l < UpperThetas.Length; l++) for (int sl = 0; sl < UpperThetas[l].Length; sl++) for (int slp1 = 0; slp1 < UpperThetas[l][sl].Length; slp1++) sum += Math.Pow(UpperThetas[l][sl][slp1], 2); return sum; }*/


                    DoFowardPropagation(ref activationUnits, upperThetas);

                    double m_ = Datas.TrainElementsCount;

                    return (1 / m_) * sumCost() + (Lambda / (2 * m_)) * sumThetaSquare();
                }


                /// <summary>
                /// Value hypothesis  
                /// </summary>
                /// <param name="activationUnit">Activation unit from a neural network</param>
                /// <param name="theta">theta values from a activation unit</param>
                /// <returns>
                /// Prediction = theta[0] * activationUnit[0] + theta[1] * activationUnit[1] + theta[2] * activationUnit[2] + ... + theta[n] * activationUnit[n]
                /// <para>Where:</para>
                /// <para>n -> Resources quantity</para>
                /// </returns>
                public double HypothesisAU(double[] theta, double[] activationUnit)
                {
                    double prediction = 0;
                    for (int n = 0; n < activationUnit.Length; n++)
                        prediction += theta[n] * activationUnit[n];
                    return prediction;
                }
                /// <summary>
                /// Value predictions  
                /// </summary>
                /// <param name="activationUnit">Activation unit from a neural network</param>
                /// <returns>
                /// Predictions = 
                /// <para>{</para>
                /// <para>    theta[l - 2][0][0] * res[0] + theta[l - 2][0][1] * res[1] + ... + theta[l - 2][0][n] * res[n],</para>
                /// <para>    theta[l - 2][1][0] * res[0] + theta[l - 2][1][1] * res[1] + ... + theta[l - 2][1][n] * res[n],</para>
                /// <para>    ...,</para>
                /// <para>    theta[l - 2][k][0] * res[0] + theta[l - 2][k][1] * res[1] + ... + theta[l - 2][k][n] * res[n]</para>
                /// <para>}</para>
                /// <para>Where:</para>
                /// <para>k -> Output classes quantity</para>
                /// <para>l -> Layers quantity</para>
                /// <para>n -> Resources quantity</para>
                /// <para>res -> Resources for final layer, after passing through all layers (excluing output layer)</para>
                /// </returns>
                public double[] PredictAU(double[] activationUnit)
                {
                    Func<double, double> aF = ActivationFunctions.GetFunction(ActiovationFunction);//Retorna uma função de ativação (Sigmoid, tangente hiperbólica, retificação linear, etc)                   
                    double[] res = activationUnit;
                    for (int l = 0; l < UpperThetas.Length; l++)
                    {
                        double[] predictions = new double[UpperThetas[l].Length];
                        double[][] weights = UpperThetas[l];
                        for (int k = 0; k < UpperThetas[l].Length; k++)
                            predictions[k] = aF(HypothesisAU(weights[k], res));
                        res = predictions;
                    }
                    return res;
                }


                /// <summary>
                /// Value hypothesis  
                /// </summary>
                /// <param name="resources">Resources from a element</param>
                /// <param name="theta">theta values from a activation unit</param>
                /// <returns>
                /// Prediction = theta[0] + theta[1] * resources[0] + theta[2] * resources[1] + ... + theta[n] * resources[n - 1]
                /// <para>Where:</para>
                /// <para>n -> Resources quantity</para>
                /// </returns>
                public double Hypothesis(double[] theta, double[] resources)
                {
                    double prediction = theta[0];
                    for (int n = 0; n < resources.Length; n++)
                        prediction += theta[n + 1] * resources[n];
                    return prediction;
                }
                /// <summary>
                /// Value predictions  
                /// </summary>
                /// <param name="resources">Resources from a element</param>
                /// <returns>
                /// Predictions = 
                /// <para>{</para>
                /// <para>    theta[l - 2][0][0] + theta[l - 2][0][1] * res[0] + ... + theta[l - 2][0][n] * res[n - 1],</para>
                /// <para>    theta[l - 2][1][0] + theta[l - 2][1][1] * res[0] + ... + theta[l - 2][1][n] * res[n - 1],</para>
                /// <para>    ...,</para>
                /// <para>    theta[l - 2][k][0] + theta[l - 2][k][1] * res[0] + ... + theta[l - 2][k][n] * res[n - 1]</para>
                /// <para>}</para>
                /// <para>Where:</para>
                /// <para>k -> Output classes quantity</para>
                /// <para>l -> Layers quantity</para>
                /// <para>n -> Resources quantity</para>
                /// <para>res -> Resources for final layer, after passing through all layers (excluing output layer)</para>
                /// </returns>
                public double[] Predict(double[] resources)
                {
                    Func<double, double> aF = ActivationFunctions.GetFunction(ActiovationFunction);//Retorna uma função de ativação (Sigmoid, tangente hiperbólica, retificação linear, etc)                   
                    double[] res = resources;
                    for (int l = 0; l < UpperThetas.Length; l++)
                    {
                        double[] predictions = new double[UpperThetas[l].Length];
                        double[][] weights = UpperThetas[l];
                        for (int k = 0; k < UpperThetas[l].Length; k++)
                            predictions[k] = aF(Hypothesis(weights[k], res));
                        res = predictions;
                    }
                    return res;
                }


                /// <summary>
                /// Neural network activations functions class
                /// </summary>
                public static class ActivationFunctions
                {
                    #region Sigmoid
                    /// <summary>
                    /// Return the sigmoid value from a informed value
                    /// </summary>
                    /// <param name="value"></param>
                    /// <returns>0 ≤ returned value ≤ 1</returns>
                    public static double Sigmoid(double value) => 1 / (1 + Math.Pow(Math.E, -value));
                    /// <summary>
                    /// Return the sigmoid values from a informed values array
                    /// </summary>
                    /// <param name="values"></param>
                    /// <returns>0 ≤ returned values ≤ 1</returns>
                    public static double[] Sigmoid(double[] values)
                    {
                        double[] sValues = new double[values.Length];
                        for (int i = 0; i < sValues.Length; i++)
                            sValues[i] = Sigmoid(values[i]);
                        return sValues;
                    }

                    /// <summary>
                    /// Return the value used on sigmoid function
                    /// </summary>
                    /// <param name="sigmoid"></param>
                    /// <returns></returns>
                    public static double InvertSigmoid(double sigmoid) => -Math.Log((1 / sigmoid) - 1);
                    /// <summary>
                    /// Return the value used on sigmoid function
                    /// </summary>
                    /// <param name="sigmoids"></param>
                    /// <returns></returns>
                    public static double[] InvertSigmoid(double[] sigmoids)
                    {
                        double[] iSValues = new double[sigmoids.Length];
                        for (int i = 0; i < iSValues.Length; i++)
                            iSValues[i] = InvertSigmoid(sigmoids[i]);
                        return iSValues;
                    }
                    #endregion

                    #region TanH
                    /// <summary>
                    /// Return the hyperbolic tangent value from a informed value
                    /// </summary>
                    /// <param name="value"></param>
                    /// <returns>-1 ≤ returned value ≤ 1</returns>
                    public static double HyperbolicTangent(double value) => (1 / (1 + Math.Pow(Math.E, -2 * value))) - 1;
                    /// <summary>
                    /// Return the hyperbolic tangent values from a informed values array
                    /// </summary>
                    /// <param name="values"></param>
                    /// <returns>-1 ≤ returned values ≤ 1</returns>
                    public static double[] HyperbolicTangent(double[] values)
                    {
                        double[] hTValues = new double[values.Length];
                        for (int i = 0; i < hTValues.Length; i++)
                            hTValues[i] = HyperbolicTangent(values[i]);
                        return hTValues;
                    }

                    /// <summary>
                    /// Return the value used on Hyperbolic Tangent function
                    /// </summary>
                    /// <param name="hyperbolicTangent"></param>
                    /// <returns></returns>
                    public static double InvertHyperbolicTangent(double hyperbolicTangent) => Math.Log((-hyperbolicTangent) / (hyperbolicTangent + 1)) / (-2);
                    /// <summary>
                    /// Return the value used on Hyperbolic Tangent function
                    /// </summary>
                    /// <param name="hyperbolicTangents"></param>
                    /// <returns></returns>
                    public static double[] InvertHyperbolicTangent(double[] hyperbolicTangents)
                    {
                        double[] iSValues = new double[hyperbolicTangents.Length];
                        for (int i = 0; i < iSValues.Length; i++)
                            iSValues[i] = InvertHyperbolicTangent(hyperbolicTangents[i]);
                        return iSValues;
                    }
                    #endregion

                    #region ReLU
                    /// <summary>
                    /// Return the linear rectifier value from a informed value
                    /// </summary>
                    /// <param name="value"></param>
                    /// <returns>
                    /// <para> 0 > returned value → 0</para>
                    /// <para> 0 ≤ returned value → returned value</para> 
                    /// </returns>
                    public static double LinearRectifier(double value) => value < 0 ? 0 : value;
                    /// <summary>
                    /// Return the linear rectifier values from a informed values array
                    /// </summary>
                    /// <param name="values"></param>
                    /// <returns>
                    /// <para> 0 > returned value → 0</para>
                    /// <para> 0 ≤ returned value → returned value</para> 
                    /// </returns>
                    public static double[] LinearRectifier(double[] values)
                    {
                        double[] sValues = new double[values.Length];
                        for (int i = 0; i < sValues.Length; i++)
                            sValues[i] = LinearRectifier(values[i]);
                        return sValues;
                    }
                    #endregion

                    #region Get Function
                    /// <summary>
                    /// Return a actiovation function who will return only one value
                    /// </summary>
                    /// <param name="function"></param>
                    /// <returns></returns>
                    public static Func<double, double> GetFunction(ActiovationFunctions function)
                    {
                        switch (function)
                        {
                            case ActiovationFunctions.Sigmoid:
                                return Sigmoid;
                            case ActiovationFunctions.TanH:
                                return HyperbolicTangent;
                            case ActiovationFunctions.ReLU:
                                return LinearRectifier;
                            default:
                                return null;
                        }
                    }
                    /// <summary>
                    /// Return a actiovation function who will return a array with values
                    /// </summary>
                    /// <param name="function"></param>
                    /// <returns></returns>
                    public static Func<double[], double[]> GetArrayFunction(ActiovationFunctions function)
                    {
                        switch (function)
                        {
                            case ActiovationFunctions.Sigmoid:
                                return Sigmoid;
                            case ActiovationFunctions.TanH:
                                return HyperbolicTangent;
                            case ActiovationFunctions.ReLU:
                                return LinearRectifier;
                            default:
                                return null;
                        }
                    }

                    /// <summary>
                    /// Return a invert actiovation function who will return only one value
                    /// </summary>
                    /// <param name="function"></param>
                    /// <returns></returns>
                    public static Func<double, double> GetInvertFunction(ActiovationFunctions function)
                    {
                        switch (function)
                        {
                            case ActiovationFunctions.Sigmoid:
                                return InvertSigmoid;
                            case ActiovationFunctions.TanH:
                                return InvertHyperbolicTangent;
                            case ActiovationFunctions.ReLU:
                                return null;
                            default:
                                return null;
                        }
                    }
                    /// <summary>
                    /// Return a invert actiovation function who will return a array with values
                    /// </summary>
                    /// <param name="function"></param>
                    /// <returns></returns>
                    public static Func<double[], double[]> GetInvertArrayFunction(ActiovationFunctions function)
                    {
                        switch (function)
                        {
                            case ActiovationFunctions.Sigmoid:
                                return Sigmoid;
                            case ActiovationFunctions.TanH:
                                return HyperbolicTangent;
                            case ActiovationFunctions.ReLU:
                                return null;
                            default:
                                return null;
                        }
                    }
                    #endregion

                    /// <summary>
                    /// Activation functions
                    /// </summary>
                    public enum ActiovationFunctions
                    {
                        /// <summary>
                        /// Sigmoid
                        /// </summary>
                        Sigmoid,
                        /// <summary>
                        /// Hyperbolic Tangent
                        /// </summary>
                        TanH,
                        /// <summary>
                        /// Linear Rectifier
                        /// </summary>
                        ReLU
                    }
                }

                //Processo de criação de Matrizes: Θ; a; δ; D; Δ.
                //Θ -> Matriz de valores Thetas
                //a -> Matriz de unidades de ativação
                //δ -> Matriz de margem de erros
                //D -> Matriz de Derivações
                //Δ -> Matriz de valores Delta
                //
                //Criando matriz Θ (double[][][])
                //Essa matriz será usado nas camadas escondidas e de saída
                //double[][][] x = new double[ΘL][][];
                //ΘL -> quantidade de camadas para Θ/ ΘL = quantidade de camadas escondidas + camada de saída (1)
                //             x[{l ∈N|l>=0^l<ΘL-1}] = new double[FR][];
                //             x[ΘL-1] = new double[OL][];
                //             FR -> quantidade de recursos + unidade de polarização (1)
                //             OL -> quantidade de respostas finais
                //                          x[{l ∈N|l>=0^l<ΘL-1}][{i ∈N|i>=0^i<FR-1}] = new double[FR];
                //                          x[ΘL-1][{i ∈N|i>=0^i<OL-1}] = new double[FR];
                //*PARA CADA ENDEREÇO DESSA MATRIZ SERÁ INICIALIZADO UM VALOR DOUBLE QUALQUER*
                //
                //Criando matriz a (double[][][]) <- ESSA OPÇÃO GERA MENOS CONSUMO DE MEMÓRIA, MAS NÃO GERA AS MATRIZES PARA CADA ELEMENTO
                //Essa matriz será usado por todas as camadas
                //double[][][] y = new double[aL][][];
                //aL -> quantidade de camadas para a/ aL = quantidade de camadas escondidas + camada de entrada (1) + camada de saída (1)
                //              y[0] = new double[FR];
                //              y[{l ∈N|l>=0^l<aL-1}] = new double[FR];
                //              y[aL-1] = new double[OL];
                //              FR -> quantidade de recursos + unidade de polarização (1)
                //              OL -> quantidade de respostas finais
                //*QUASE TODOS OS ENDEREÇOS TERÃO O VALOR 0, PORÉM, PARA CADA MATRIZ DE TAMANHO 'FR', NO ENDEREÇO 0, SERÁ ATRIBUIDO O VALOR 1*
                //
                //Criando matriz a (double[][][]) <- ESSA OPÇÃO GERA UM CONSUMO MUITO MAIOR DE MEMÓRIA
                //Essa matriz será usado por todas as camadas
                //double[][][] y = new double[aL][][];
                //aL -> quantidade de camadas para a/ aL = quantidade de camadas escondidas + camada de entrada (1) + camada de saída (1)
                //             y[{l ∈N|l>=0^l<aL-1}] = new double[m][];
                //             m -> quantidade de elementos
                //                 y[][0] = new double[FR];
                //                 y[][{i ∈N|i>0^i<m-1}] = new double[FR];
                //                 y[][m-1] = new double[OL];
                //                 FR -> quantidade de recursos + unidade de polarização (1)
                //                 OL -> quantidade de respostas finais
                //*QUASE TODOS OS ENDEREÇOS TERÃO O VALOR 0, PORÉM, PARA CADA MATRIZ DE TAMANHO 'FR', NO ENDEREÇO 0, SERÁ ATRIBUIDO O VALOR 1*
                //
                //Criando matriz δ (double[][])
                //Essa matriz será usado nas camadas escondidas e de saída
                //double[][] z = new double[δL][];
                //δL -> quantidade de camadas para δ/ δL = quantidade de camadas escondidas + camada de entrada (1)
                //             z[{l ∈N|l>=0^l<δL-1}] = new double[FR];
                //             z[δL-1] = new double[OL];
                //             FR -> quantidade de recursos + unidade de polarização (1)
                //             OL -> quantidade de respostas finais
                //*PARA CADA ENDEREÇO, SE DEFINE O VALOR 0*
                //
                //Criando matriz D (double[][][])
                //Essa matriz será usado nas camadas escondidas e de saída
                //double[][][] v = new double[DL][][];
                //DL -> quantidade de camadas para D/ DL = quantidade de camadas escondidas + camada de saída (1)
                //             v[{l ∈N|l>=0^l<DL-1}] = new double[FR][];
                //             v[DL-1] = new double[OL][];
                //             FR -> quantidade de recursos + unidade de polarização (1)
                //             OL -> quantidade de respostas finais
                //                          v[{l ∈N|l>=0^l<DL-1}][{i ∈N|i>=0^i<FR-1}] = new double[FR];
                //                          v[DL-1][{i ∈N|i>=0^i<OL-1}] = new double[FR];
                //*PARA CADA ENDEREÇO DESSA MATRIZ SERÁ DEFINIDO O VALOR 0*
                //
                //Criando matriz Δ (double[][][])
                //Essa matriz será usado nas camadas escondidas e de saída
                //double[][][] w = new double[ΔL][][];
                //ΔL -> quantidade de camadas para Δ/ ΔL = quantidade de camadas escondidas + camada de saída (1)
                //             v[{l ∈N|l>=0^l<ΔL-1}] = new double[FR][];
                //             v[ΔL-1] = new double[OL][];
                //             FR -> quantidade de recursos + unidade de polarização (1)
                //             OL -> quantidade de respostas finais
                //                          v[{l ∈N|l>=0^l<ΔL-1}][{i ∈N|i>=0^i<FR-1}] = new double[FR];
                //                          v[ΔL-1][{i ∈N|i>=0^i<OL-1}] = new double[FR];
                //*PARA CADA ENDEREÇO DESSA MATRIZ SERÁ DEFINIDO O VALOR 0*
                //
                //
                //
                //















                //Processos de fowardpropagation e backpropagation
                //
                //Duas estapas devem ser feitas para criação da função de custo utilizando redes neurais:
                //*Fowardpropagation -> o mesmo processo feito na regressão linear ou regressão logistica
                //*Backwardpropagation -> obtem a margem de erro e retornar as camadas para recomeçar o processo de fowardpropagation
                //
                //Símbolos:
                //∑ {índice final|váriavel = índice inicial} <- Somatória
                //log <- logarítmo (base 2)
                //m <- quantidade de elementos
                //n <- quantidade de recursos
                //k <- quantidade de camadas
                //α <- alpha (constante de "passo" do gradiente descendente)
                //λ <- lambda (constante de reajuste da função de custo)
                //Θ <- teta (Maiúscula) (matriz contendos todos os valores peso das funções de custo)
                //θ <- teta (Minúscula) (valores peso da função de custo)
                //Sl <- quantide de unidades (s/ contar a unidade de polarização (aquela que vale 1)) na camada l (pela quantidade de camadas 'k')
                //array[1º;2º;3º;...] <- matriz qualquer possuindo 3 dimensões de endereço (Ex.: ∑{a|i = 0} ∑{b|j = 0} ∑{c|k = 0} array[i;j;k])
                //z[l] <- hipótese hθ(x) = θ0*x0 + θ1*x1 + θ2*x2 + ... + θj*xj
                //g(z[l]) <- função sigmoid
                //δ <- erro do nó 'j' (pela quantidade de recursos 'n'), na camada 'l' (pela quantidade de camadas 'k')
                //Δ[l;i;j] <- Somatória do produto da unidade de ativação pelo erro δ (camada l; elemento i; recurso j)
                //J(Θ) <- Função de custo
                //D[l;i;j] = (∂/∂Θ[l;i;j])J(Θ) = min J(Θ) <- Função de custo minimizado
                //a[l;i;j] <- Unidade de ativação
                //
                //
                //
                //
                //Formulas:
                //* J(Θ) = (-1/m)[ ∑{m|i=1} ∑{K|k=1} y[i;k]*log(hθ(x[i])[k]) + (1-y[i;k])*log(1-hθ(x[i])[k])] + (λ/(2*m))*∑{L-1|l=1} ∑{Sl|i=1}  ∑{Sl+1|j=1} (Θ[l;j])²
                //* a[0] = X <- A unidade de ativação da camada 0 é igual ao valores de treino X
                //* a[{l ∈N|l>0}] = g(z[{l ∈N|l>0}])
                //* δ[k] = a[k;j] - y[j] <- margem de erro para a ultima camada, onde será comparado com o valor final
                //* δ[{l ∈N|l>0 ^ l<k}] = (Θ[{l ∈N|l>0 ^ l<k}]) Transposto .* g'(z[{l ∈N|l>0 ^ l<k}])
                //* g'(z[{l ∈N|l>0 ^ l<k}]) = a[{l ∈N|l>0 ^ l<k}] .* (1 - a[{l ∈N|l>0 ^ l<k}])
                //* Δ[l;i;j] = Δ[l;i;j] + a[l;j] * δ[l+1;j]
                //* Δ[l] = Δ[l] + δ[l+1;j] * (a[l;j]) Transposto
                //* D[l;i;j] = 1/m * Δ[l;i;j] + λ * Θ[l;i;j] <-> j ≠ 0
                //* D[l;i;j] = 1/m * Δ[l;i;j] <-> j = 0
                //* (∂/∂Θ[l;i;j])J(Θ) = D[l;i;j]
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
                    double[][] normalizedArray = new double[array.Length][];
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
                                if (double.IsNaN(mM[n][0]) || array[m][n] > mM[n][0])
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
                                if (mM[j] == null)
                                    mM[j] = new double[2] { double.NaN, double.NaN };
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
                            for (int j = 0; j < array[i].Length; j++)
                                array[i][j] = rnd.NextDouble();
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                            for (int j = 0; j < array[i].Length; j++)
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
                            for (int j = 0; j < array[i].Length; j++)
                                for (int k = 0; k < array[i][j].Length; k++)
                                    array[i][j][k] = rnd.NextDouble();
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                            for (int j = 0; j < array[i].Length; j++)
                                for (int k = 0; k < array[i][j].Length; k++)
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
                            for (int j = 0; j < array[i].Length; j++)
                                newArray[i][j] = rnd.NextDouble();
                        }
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                        {
                            newArray[i] = new double[array[i].Length];
                            for (int j = 0; j < array[i].Length; j++)
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
                            for (int j = 0; j < array[i].Length; j++)
                            {
                                newArray[i][j] = new double[array[i][j].Length];
                                for (int k = 0; k < array[i][j].Length; k++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (int i = 0; i < array.Length; i++)
                        {
                            newArray[i] = new double[array[i].Length][];
                            for (int j = 0; j < array[i].Length; j++)
                            {
                                newArray[i][j] = new double[array[i][j].Length];
                                for (int k = 0; k < array[i][j].Length; k++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return newArray;
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
                    return newArray;
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
                    return newArray;
                }



                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static void FillArray(ref double[] array, double value)
                {

                    for (int i = 0; i < array.Length; i++)
                        array[i] = value;

                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static void FillArray(ref double[][] array, double value)
                {

                    for (int i = 0; i < array.Length; i++)
                        for (int j = 0; j < array[i].Length; j++)
                            array[i][j] = value;

                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static void FillArray(ref double[][][] array, double value)
                {

                    for (int i = 0; i < array.Length; i++)
                        for (int j = 0; j < array[i].Length; j++)
                            for (int k = 0; k < array[i][j].Length; k++)
                                array[i][j][k] = value;

                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static void CUDAFillArray(ref double[] array, double value)
                {
                    double[] filleArray = array;

                    Parallel.For(0, array.Length, i => { filleArray[i] = value; });
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static void CUDAFillArray(ref double[][] array, double value)
                {
                    double[][] filleArray = array;
                    Parallel.For(0, array.Length, i => { Parallel.For(0, filleArray[i].Length, j => { filleArray[i][j] = value; }); });
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static void CUDAFillArray(ref double[][][] array, double value)
                {
                    double[][][] filleArray = array;
                    Parallel.For(0, array.Length, i => { Parallel.For(0, filleArray[i].Length, j => { Parallel.For(0, filleArray[i][j].Length, k => { filleArray[i][j][k] = value; }); }); });
                }



                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static double[] FillArray(double[] array, double value)
                {
                    double[] newArray = new double[array.Length];


                    for (int i = 0; i < array.Length; i++)
                        newArray[i] = value;

                    return newArray;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] FillArray(double[][] array, double value)
                {
                    double[][] newArray = new double[array.Length][];

                    for (int i = 0; i < array.Length; i++)
                    {
                        newArray[i] = new double[array[i].Length];
                        for (int j = 0; j < array[i].Length; j++)
                            newArray[i][j] = value;
                    }

                    return newArray;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static double[][][] FillArray(double[][][] array, double value)
                {
                    double[][][] newArray = new double[array.Length][][];

                    for (int i = 0; i < array.Length; i++)
                    {
                        newArray[i] = new double[array[i].Length][];
                        for (int j = 0; j < array[i].Length; j++)
                        {
                            newArray[i][j] = new double[array[i][j].Length];
                            for (int k = 0; k < array[i][j].Length; k++)
                                newArray[i][j][k] = value;
                        }
                    }

                    return newArray;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static double[] CUDAFillArray(double[] array, double value)
                {
                    double[] newArray = new double[array.Length];

                    Parallel.For(0, array.Length, i => { newArray[i] = value; });
                    return newArray;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] CUDAFillArray(double[][] array, double value)
                {
                    double[][] newArray = new double[array.Length][];
                    Parallel.For(0, array.Length, i => { newArray[i] = new double[array[i].Length]; Parallel.For(0, array[i].Length, j => { newArray[i][j] = value; }); });
                    return newArray;
                }
                /// <summary>
                /// Fill a array with double values
                /// </summary>            
                /// <param name="array">double type array</param>
                /// <param name="value">Value to set on array</param>
                public static double[][][] CUDAFillArray(double[][][] array, double value)
                {
                    double[][][] newArray = new double[array.Length][][];

                    Parallel.For(0, array.Length, i => { newArray[i] = new double[array[i].Length][]; Parallel.For(0, array[i].Length, j => { newArray[i][j] = new double[array[i][j].Length]; Parallel.For(0, array[i][j].Length, k => { newArray[i][j][k] = value; }); }); });
                    return newArray;
                }
                #endregion

                #region Creat and Fill Array
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[] CreatFillArray(int l, bool setRandomValues = false)
                {
                    double[] newArray = new double[l];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < l; i++)
                            newArray[i] = rnd.NextDouble();
                    }
                    else
                        for (int i = 0; i < l; i++)
                            newArray[i] = 0;
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] CreatFillArray(int l, int m, bool setRandomValues = false)
                {
                    double[][] newArray = new double[l][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m];
                            for (int j = 0; j < m; j++)
                                newArray[i][j] = rnd.NextDouble();
                        }
                    }
                    else
                    {
                        for (int i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m];
                            for (int j = 0; j < m; j++)
                                newArray[i][j] = 0;
                        }
                    }
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="n">3° array lenght</param>           
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CreatFillArray(int l, int m, int n, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m][];
                            for (int j = 0; j < m; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (int k = 0; k < n; k++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (uint i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m][];
                            for (uint j = 0; j < m; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (uint k = 0; k < n; k++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return newArray;
                }


                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[] CreatFillArray(long l, bool setRandomValues = false)
                {
                    double[] newArray = new double[l];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (long i = 0; i < l; i++)
                            newArray[i] = rnd.NextDouble();
                    }
                    else
                        for (long i = 0; i < l; i++)
                            newArray[i] = 0;
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] CreatFillArray(long l, long m, bool setRandomValues = false)
                {
                    double[][] newArray = new double[l][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (long i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m];
                            for (long j = 0; j < m; j++)
                                newArray[i][j] = rnd.NextDouble();
                        }
                    }
                    else
                    {
                        for (long i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m];
                            for (long j = 0; j < m; j++)
                                newArray[i][j] = 0;
                        }
                    }
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="n">3° array lenght</param>           
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CreatFillArray(long l, long m, long n, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (long i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m][];
                            for (long j = 0; j < m; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (long k = 0; k < n; k++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (long i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m][];
                            for (long j = 0; j < m; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (long k = 0; k < n; k++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return newArray;
                }


                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[] CreatFillArray(ulong l, bool setRandomValues = false)
                {
                    double[] newArray = new double[l];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (ulong i = 0; i < l; i++)
                            newArray[i] = rnd.NextDouble();
                    }
                    else
                        for (ulong i = 0; i < l; i++)
                            newArray[i] = 0;
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] CreatFillArray(ulong l, ulong m, bool setRandomValues = false)
                {
                    double[][] newArray = new double[l][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (ulong i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m];
                            for (ulong j = 0; j < m; j++)
                                newArray[i][j] = rnd.NextDouble();
                        }
                    }
                    else
                    {
                        for (ulong i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m];
                            for (ulong j = 0; j < m; j++)
                                newArray[i][j] = 0;
                        }
                    }
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="n">3° array lenght</param>           
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CreatFillArray(ulong l, ulong m, ulong n, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (ulong i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m][];
                            for (ulong j = 0; j < m; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (ulong k = 0; k < n; k++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (ulong i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m][];
                            for (ulong j = 0; j < m; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (ulong k = 0; k < n; k++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return newArray;
                }


                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[] CUDACreatFillArray(int l, bool setRandomValues = false)
                {
                    double[] newArray = new double[l];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, l, i => { newArray[i] = rnd.NextDouble(); });
                    }
                    else
                        Parallel.For(0, l, i => { newArray[i] = 0; });
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] CUDACreatFillArray(int l, int m, bool setRandomValues = false)
                {
                    double[][] newArray = new double[l][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, l, i => { newArray[i] = new double[m]; Parallel.For(0, m, j => { newArray[i][j] = rnd.NextDouble(); }); });
                    }
                    else
                        Parallel.For(0, l, i => { newArray[i] = new double[m]; Parallel.For(0, m, j => { newArray[i][j] = 0; }); });
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="n">3° array lenght</param>  
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CUDACreatFillArray(int l, int m, int n, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, l, i => { newArray[i] = new double[m][]; Parallel.For(0, m, j => { newArray[i][j] = new double[n]; Parallel.For(0, n, k => { newArray[i][j][k] = rnd.NextDouble(); }); }); });
                    }
                    else
                        Parallel.For(0, l, i => { newArray[i] = new double[m][]; Parallel.For(0, m, j => { newArray[i][j] = new double[n]; Parallel.For(0, n, k => { newArray[i][j][k] = 0; }); }); });
                    return newArray;
                }




                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">Array lenghts</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] CreatFillArray(int[] l, bool setRandomValues = false)
                {
                    double[][] newArray = new double[l.Length][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < l.Length; i++)
                        {
                            newArray[i] = new double[l[i]];
                            for (int j = 0; j < l[i]; j++)
                                newArray[i][j] = rnd.NextDouble();
                        }
                    }
                    else
                    {
                        for (int i = 0; i < l.Length; i++)
                        {
                            newArray[i] = new double[l[i]];
                            for (int j = 0; j < l[i]; j++)
                                newArray[i][j] = 0;
                        }
                    }
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>        
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CreatFillArray(int l, int[] m, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m.Length][];
                            for (int j = 0; j < m.Length; j++)
                            {
                                newArray[i][j] = new double[m[i]];
                                for (int k = 0; k < m[i]; k++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (uint i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m.Length][];
                            for (uint j = 0; j < m.Length; j++)
                            {
                                newArray[i][j] = new double[m[i]];
                                for (uint k = 0; k < m[i]; k++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="n">3° array lenght</param>        
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CreatFillArray(int[] l, int n, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l.Length][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (int i = 0; i < l.Length; i++)
                        {
                            newArray[i] = new double[l[i]][];
                            for (int j = 0; j < l[i]; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (int k = 0; k < n; k++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (int i = 0; i < l.Length; i++)
                        {
                            newArray[i] = new double[l[i]][];
                            for (int j = 0; j < l[i]; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (int k = 0; k < n; k++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return newArray;
                }


                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">Array lenghts</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] CreatFillArray(long[] l, bool setRandomValues = false)
                {
                    double[][] newArray = new double[l.Length][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (long i = 0; i < l.Length; i++)
                        {
                            newArray[i] = new double[l[i]];
                            for (long j = 0; j < l[i]; j++)
                                newArray[i][j] = rnd.NextDouble();
                        }
                    }
                    else
                    {
                        for (long i = 0; i < l.Length; i++)
                        {
                            newArray[i] = new double[l[i]];
                            for (long j = 0; j < l[i]; j++)
                                newArray[i][j] = 0;
                        }
                    }
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>        
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CreatFillArray(long l, long[] m, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (long i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m.Length][];
                            for (long j = 0; j < m.Length; j++)
                            {
                                newArray[i][j] = new double[m[i]];
                                for (long k = 0; k < m[i]; k++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (uint i = 0; i < l; i++)
                        {
                            newArray[i] = new double[m.Length][];
                            for (uint j = 0; j < m.Length; j++)
                            {
                                newArray[i][j] = new double[m[i]];
                                for (uint k = 0; k < m[i]; k++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="n">3° array lenght</param>        
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CreatFillArray(long[] l, long n, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l.Length][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        for (long i = 0; i < l.Length; i++)
                        {
                            newArray[i] = new double[l[i]][];
                            for (long j = 0; j < l[i]; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (long k = 0; k < n; k++)
                                    newArray[i][j][k] = rnd.NextDouble();
                            }
                        }
                    }
                    else
                    {
                        for (long i = 0; i < l.Length; i++)
                        {
                            newArray[i] = new double[l[i]][];
                            for (long j = 0; j < l[i]; j++)
                            {
                                newArray[i][j] = new double[n];
                                for (long k = 0; k < n; k++)
                                    newArray[i][j][k] = 0;
                            }
                        }
                    }
                    return newArray;
                }

                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][] CUDACreatFillArray(int[] l, bool setRandomValues = false)
                {
                    double[][] newArray = new double[l.Length][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, l.Length, i => { newArray[i] = new double[l[i]]; Parallel.For(0, l[i], j => { newArray[i][j] = rnd.NextDouble(); }); });
                    }
                    else
                        Parallel.For(0, l.Length, i => { newArray[i] = new double[l[i]]; Parallel.For(0, l[i], j => { newArray[i][j] = 0; }); });
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>  
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CUDACreatFillArray(int l, int[] m, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, l, i => { newArray[i] = new double[m.Length][]; Parallel.For(0, m.Length, j => { newArray[i][j] = new double[m[i]]; Parallel.For(0, m[i], k => { newArray[i][j][k] = rnd.NextDouble(); }); }); });
                    }
                    else
                        Parallel.For(0, l, i => { newArray[i] = new double[m.Length][]; Parallel.For(0, m.Length, j => { newArray[i][j] = new double[m[i]]; Parallel.For(0, m[i], k => { newArray[i][j][k] = 0; }); }); });
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="n">3° array lenght</param>        
                /// <param name="setRandomValues">True: Set random values on array index/ False: Set 0 on array index</param>
                public static double[][][] CUDACreatFillArray(int[] l, int n, bool setRandomValues = false)
                {
                    double[][][] newArray = new double[l.Length][][];
                    if (setRandomValues)
                    {
                        Random rnd = new Random(DateTime.Now.Millisecond);
                        Parallel.For(0, l.Length, i => { newArray[i] = new double[l[i]][]; Parallel.For(0, l[i], j => { newArray[i][j] = new double[n]; Parallel.For(0, n, k => { newArray[i][j][k] = rnd.NextDouble(); }); }); });
                    }
                    else
                        Parallel.For(0, l.Length, i => { newArray[i] = new double[l[i]][]; Parallel.For(0, l[i], j => { newArray[i][j] = new double[n]; Parallel.For(0, n, k => { newArray[i][j][k] = 0; }); }); });
                    return newArray;
                }










                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="value">Value to set on array</param>                
                public static double[] CreatFillArray(int l, double value)
                {
                    double[] newArray = new double[l];
                    for (int i = 0; i < l; i++)
                        newArray[i] = value;
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] CreatFillArray(int l, int m, double value)
                {
                    double[][] newArray = new double[l][];

                    for (int i = 0; i < l; i++)
                    {
                        newArray[i] = new double[m];
                        for (int j = 0; j < m; j++)
                            newArray[i][j] = value;
                    }

                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="n">3° array lenght</param>           
                /// <param name="value">Value to set on array</param>
                public static double[][][] CreatFillArray(int l, int m, int n, double value)
                {
                    double[][][] newArray = new double[l][][];

                    for (uint i = 0; i < l; i++)
                    {
                        newArray[i] = new double[m][];
                        for (uint j = 0; j < m; j++)
                        {
                            newArray[i][j] = new double[n];
                            for (uint k = 0; k < n; k++)
                                newArray[i][j][k] = value;
                        }
                    }

                    return newArray;
                }


                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="value">Value to set on array</param>
                public static double[] CreatFillArray(long l, double value)
                {
                    double[] newArray = new double[l];

                    for (long i = 0; i < l; i++)
                        newArray[i] = value;
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] CreatFillArray(long l, long m, double value)
                {
                    double[][] newArray = new double[l][];

                    for (long i = 0; i < l; i++)
                    {
                        newArray[i] = new double[m];
                        for (long j = 0; j < m; j++)
                            newArray[i][j] = value;
                    }

                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="n">3° array lenght</param>           
                /// <param name="value">Value to set on array</param>
                public static double[][][] CreatFillArray(long l, long m, long n, double value)
                {
                    double[][][] newArray = new double[l][][];

                    for (long i = 0; i < l; i++)
                    {
                        newArray[i] = new double[m][];
                        for (long j = 0; j < m; j++)
                        {
                            newArray[i][j] = new double[n];
                            for (long k = 0; k < n; k++)
                                newArray[i][j][k] = value;
                        }
                    }

                    return newArray;
                }


                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="value">Value to set on array</param>
                public static double[] CreatFillArray(ulong l, double value)
                {
                    double[] newArray = new double[l];

                    for (ulong i = 0; i < l; i++)
                        newArray[i] = value;
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] CreatFillArray(ulong l, ulong m, double value)
                {
                    double[][] newArray = new double[l][];

                    for (ulong i = 0; i < l; i++)
                    {
                        newArray[i] = new double[m];
                        for (ulong j = 0; j < m; j++)
                            newArray[i][j] = value;
                    }

                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="n">3° array lenght</param>           
                /// <param name="value">Value to set on array</param>
                public static double[][][] CreatFillArray(ulong l, ulong m, ulong n, double value)
                {
                    double[][][] newArray = new double[l][][];

                    for (ulong i = 0; i < l; i++)
                    {
                        newArray[i] = new double[m][];
                        for (ulong j = 0; j < m; j++)
                        {
                            newArray[i][j] = new double[n];
                            for (ulong k = 0; k < n; k++)
                                newArray[i][j][k] = value;
                        }
                    }

                    return newArray;
                }


                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="value">Value to set on array</param>
                public static double[] CUDACreatFillArray(int l, double value)
                {
                    double[] newArray = new double[l];

                    Parallel.For(0, l, i => { newArray[i] = value; });
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] CUDACreatFillArray(int l, int m, double value)
                {
                    double[][] newArray = new double[l][];

                    Parallel.For(0, l, i => { newArray[i] = new double[m]; Parallel.For(0, m, j => { newArray[i][j] = value; }); });
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>
                /// <param name="n">3° array lenght</param>  
                /// <param name="value">Value to set on array</param>
                public static double[][][] CUDACreatFillArray(int l, int m, int n, double value)
                {
                    double[][][] newArray = new double[l][][];

                    Parallel.For(0, l, i => { newArray[i] = new double[m][]; Parallel.For(0, m, j => { newArray[i][j] = new double[n]; Parallel.For(0, n, k => { newArray[i][j][k] = value; }); }); });
                    return newArray;
                }




                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">Array lenghts</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] CreatFillArray(int[] l, double value)
                {
                    double[][] newArray = new double[l.Length][];

                    for (int i = 0; i < l.Length; i++)
                    {
                        newArray[i] = new double[l[i]];
                        for (int j = 0; j < l[i]; j++)
                            newArray[i][j] = value;
                    }

                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>        
                /// <param name="value">Value to set on array</param>
                public static double[][][] CreatFillArray(int l, int[] m, double value)
                {
                    double[][][] newArray = new double[l][][];

                    for (uint i = 0; i < l; i++)
                    {
                        newArray[i] = new double[m.Length][];
                        for (uint j = 0; j < m.Length; j++)
                        {
                            newArray[i][j] = new double[m[i]];
                            for (uint k = 0; k < m[i]; k++)
                                newArray[i][j][k] = value;
                        }
                    }

                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="n">3° array lenght</param>        
                /// <param name="value">Value to set on array</param>
                public static double[][][] CreatFillArray(int[] l, int n, double value)
                {
                    double[][][] newArray = new double[l.Length][][];

                    for (int i = 0; i < l.Length; i++)
                    {
                        newArray[i] = new double[l[i]][];
                        for (int j = 0; j < l[i]; j++)
                        {
                            newArray[i][j] = new double[n];
                            for (int k = 0; k < n; k++)
                                newArray[i][j][k] = value;
                        }
                    }

                    return newArray;
                }

                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>            
                /// <param name="l">Array lenghts</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] CreatFillArray(long[] l, double value)
                {
                    double[][] newArray = new double[l.Length][];

                    for (long i = 0; i < l.Length; i++)
                    {
                        newArray[i] = new double[l[i]];
                        for (long j = 0; j < l[i]; j++)
                            newArray[i][j] = value;
                    }

                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>        
                /// <param name="value">Value to set on array</param>
                public static double[][][] CreatFillArray(long l, long[] m, double value)
                {
                    double[][][] newArray = new double[l][][];

                    for (uint i = 0; i < l; i++)
                    {
                        newArray[i] = new double[m.Length][];
                        for (uint j = 0; j < m.Length; j++)
                        {
                            newArray[i][j] = new double[m[i]];
                            for (uint k = 0; k < m[i]; k++)
                                newArray[i][j][k] = value;
                        }
                    }

                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="n">3° array lenght</param>        
                /// <param name="value">Value to set on array</param>
                public static double[][][] CreatFillArray(long[] l, long n, double value)
                {
                    double[][][] newArray = new double[l.Length][][];

                    for (long i = 0; i < l.Length; i++)
                    {
                        newArray[i] = new double[l[i]][];
                        for (long j = 0; j < l[i]; j++)
                        {
                            newArray[i][j] = new double[n];
                            for (long k = 0; k < n; k++)
                                newArray[i][j][k] = value;
                        }
                    }

                    return newArray;
                }

                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="value">Value to set on array</param>
                public static double[][] CUDACreatFillArray(int[] l, double value)
                {
                    double[][] newArray = new double[l.Length][];

                    Parallel.For(0, l.Length, i => { newArray[i] = new double[l[i]]; Parallel.For(0, l[i], j => { newArray[i][j] = value; }); });
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="m">2° array lenght</param>  
                /// <param name="value">Value to set on array</param>
                public static double[][][] CUDACreatFillArray(int l, int[] m, double value)
                {
                    double[][][] newArray = new double[l][][];

                    Parallel.For(0, l, i => { newArray[i] = new double[m.Length][]; Parallel.For(0, m.Length, j => { newArray[i][j] = new double[m[i]]; Parallel.For(0, m[i], k => { newArray[i][j][k] = value; }); }); });
                    return newArray;
                }
                /// <summary>
                /// Creat and fill a array with double values
                /// </summary>
                /// <param name="l">1° array lenght</param>
                /// <param name="n">3° array lenght</param>        
                /// <param name="value">Value to set on array</param>
                public static double[][][] CUDACreatFillArray(int[] l, int n, double value)
                {
                    double[][][] newArray = new double[l.Length][][];

                    Parallel.For(0, l.Length, i => { newArray[i] = new double[l[i]][]; Parallel.For(0, l[i], j => { newArray[i][j] = new double[n]; Parallel.For(0, n, k => { newArray[i][j][k] = value; }); }); });
                    return newArray;
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

                    for (int i = 0; i < jaggedArray.Length; i++)
                        if (tJALenght != jaggedArray[i].Length) throw new ArgumentException("Jagged array with irregular array sizes");

                    double[][] tJaggedArray = new double[tJALenght][];
                    for (int i = 0; i < tJALenght; i++)
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

                #region Multiply
                /// <summary>
                /// Multiply two jagged array
                /// <para>Allowed operation:</para>
                /// <para>a [m][n] x b [o][p]</para>
                /// <para>n == o → c [m][p]</para>
                /// <para>Where:</para>
                /// <para>a -> 1º Jagged Array</para>
                /// <para>b -> 2º Jagged Array </para>
                /// <para>c -> Result of the multiplication</para>
                /// <para>m -> 'a' jagged array lenght</para>
                /// <para>n -> 'a' arrays lenght</para>
                /// <para>o -> 'b' jagged array lenght</para>
                /// <para>p -> 'b' arrays lenght</para>
                /// </summary>
                /// <param name="a">Jagged array</param>
                /// <param name="b">Jagged array</param>
                /// <returns></returns>
                public static double[][] Multiply(double[][] a, double[][] b)
                {
                    int
                            am = a.Length,
                            an = a.First().Length,
                            bm = b.Length,
                            bn = b.First().Length;

                    if (an != bm)
                        return null;

                    double[][] c = new double[am][];
                    for (int i = 0; i < am; i++)
                    {
                        c[i] = new double[bn];
                        for (int j = 0; j < bn; j++)
                        {
                            c[i][j] = 0;
                            for (int k = 0; k < an; k++)
                                c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                    return c;
                }
                /// <summary>
                /// Multiply 1 jagged array and 1 array
                /// <para>Allowed operation:</para>
                /// <para>a [m][n] x b [o]</para>
                /// <para>n == o → c [m]</para>
                /// <para>Where:</para>
                /// <para>a -> 1º Jagged Array</para>
                /// <para>b -> 2º Jagged Array </para>
                /// <para>c -> Result of the multiplication</para>
                /// <para>m -> 'a' jagged array lenght</para>
                /// <para>n -> 'a' arrays lenght</para>
                /// <para>o -> 'b' array lenght</para>
                /// </summary>
                /// <param name="a">Jagged array</param>
                /// <param name="b">Jagged array</param>
                /// <returns></returns>
                public static double[] Multiply(double[][] a, double[] b)
                {
                    int
                            am = a.Length,
                            an = a.First().Length,
                            bm = b.Length;
                    if (an != bm)
                        return null;
                    double[] c = new double[am];
                    for (int i = 0; i < am; i++)
                    {
                        c[i] = 0;
                        for (int k = 0; k < an; k++)
                            c[i] += a[i][k] * b[k];
                    }
                    return c;
                }
                #endregion

                #region Sum
                /// <summary>
                /// Sum 1 or more jagged arrays
                /// *Warning: Every jagged array must have the same size
                /// </summary>
                /// <param name="jaggedArrays"></param>
                /// <returns></returns>
                public static double[][] Sum(params double[][][] jaggedArrays)
                {                    
                    double[][] a = jaggedArrays.First();
                    for (int i = 1; i < jaggedArrays.Length; i++)
                        for (int j = 0; j < jaggedArrays[i].Length; j++)
                            for (int k = 0; k < jaggedArrays[i][j].Length; k++)
                                a[j][k] += jaggedArrays[i][j][k];
                    return a;
                }
                /// <summary>
                /// Sum 1 or more arrays
                /// *Warning: Every array must have the same size
                /// </summary>
                /// <param name="arrays"></param>
                /// <returns></returns>
                public static double[] Sum(params double[][] arrays)
                {
                    double[] a = arrays.First();
                    for (int i = 1; i < arrays.Length; i++)
                        for (int j = 0; j < arrays[i].Length; j++)
                                a[j] += arrays[i][j];
                    return a;
                }
                #endregion

                #region Subtraction
                /// <summary>
                /// Subtract 1 or more jagged arrays
                /// *Warning: Every jagged array must have the same size
                /// </summary>
                /// <param name="jaggedArrays"></param>
                /// <returns></returns>
                public static double[][] Subtract(params double[][][] jaggedArrays)
                {
                    double[][] a = jaggedArrays.First();
                    for (int i = 1; i < jaggedArrays.Length; i++)
                        for (int j = 0; j < jaggedArrays[i].Length; j++)
                            for (int k = 0; k < jaggedArrays[i][j].Length; k++)
                                a[j][k] -= jaggedArrays[i][j][k];
                    return a;
                }
                /// <summary>
                /// Subtract 1 or more arrays
                /// *Warning: Every array must have the same size
                /// </summary>
                /// <param name="arrays"></param>
                /// <returns></returns>
                public static double[] Subtract(params double[][] arrays)
                {
                    double[] a = arrays.First();
                    for (int i = 1; i < arrays.Length; i++)
                        for (int j = 0; j < arrays[i].Length; j++)
                            a[j] -= arrays[i][j];
                    return a;
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



        /// <summary>
        /// 
        /// </summary>
        public static class Gt
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="array"></param>
            /// <returns></returns>
            public static string WriteArray(double[] array)
            {
                string t = "{";
                foreach (double x in array)
                    t += t == "{" ? $"{x}" : $";{x}";
                t += "}";
                return t;
            }
            /// <summary>
            /// 
            /// </summary>
            /// <param name="array"></param>
            /// <param name="previousText"></param>
            /// <returns></returns>
            public static string WriteArray(double[][] array, string previousText = "")
            {
                string t = string.Empty;
                for (int i = 0; i < array.Length; i++)
                    t += string.IsNullOrEmpty(t) ? $"{previousText}[{i}]{WriteArray(array[i])}" : $"\n{previousText}[{i}]{WriteArray(array[i])}";
                return t;
            }
            /// <summary>
            /// 
            /// </summary>
            /// <param name="array"></param>
            /// <returns></returns>
            public static string WriteArray(double[][][] array)
            {
                string t = string.Empty;
                for (int i = 0; i < array.Length; i++)
                    t += string.IsNullOrEmpty(t) ? $"{WriteArray(array[i], $"[{i}]")}" : $"\n{WriteArray(array[i], $"[{i}]")}";
                return t;
            }
            /// <summary>
            /// 
            /// </summary>
            /// <param name="array"></param>
            /// <returns></returns>
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




