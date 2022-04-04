namespace CUDA_ML_Libary
{
    /// <summary>
    /// CUDA Machine Learning Neural Network tuple
    /// </summary>
    /// <typeparam name="T1"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.UpperThetas"/></typeparam>
    /// <typeparam name="T2"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.UpperThetasTotalLenght"/></typeparam>
    /// <typeparam name="T3"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.Derivatives"/></typeparam>
    /// <typeparam name="T4"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.DerivativesTotalLenght"/></typeparam>
    /// <typeparam name="T5"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.Deltas"/></typeparam>
    /// <typeparam name="T6"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.DeltasTotalLenght"/></typeparam>
    /// <typeparam name="T7"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.ErrorMargins"/></typeparam>
    /// <typeparam name="T8"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.ErrorMarginsTotalLenght"/></typeparam>
    /// <typeparam name="T9"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.ActivationUnits"/></typeparam>
    /// <typeparam name="T10"><see cref="MachineLearning.SupervisedLearning.NeuralNetworks.ActivationUnitsTotalLenght"/></typeparam>
    public class Tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>
    {
        /// <summary>
        /// 
        /// </summary>
        public double[][][] upperThetas;
        /// <summary>
        /// 
        /// </summary>
        public double upperThetasTotalLenght;
        /// <summary>
        /// 
        /// </summary>
        public double[][][] derivatives;
        /// <summary>
        /// 
        /// </summary>
        public double derivativesTotalLenght;
        /// <summary>
        /// 
        /// </summary>
        public double[][][] deltas;
        /// <summary>
        /// 
        /// </summary>
        public double deltasTotalLenght;
        /// <summary>
        /// 
        /// </summary>
        public double[][] errorMargins;
        /// <summary>
        /// 
        /// </summary>
        public double errorMarginsTotalLenght;
        /// <summary>
        /// 
        /// </summary>
        public double[][][] activationUnits;
        /// <summary>
        /// 
        /// </summary>
        public double activationUnitsTotalLenght;

        /// <summary>
        /// Creat new tuple instance
        /// </summary>
        /// <param name="upperThetas"></param>
        /// <param name="upperThetasTotalLenght"></param>
        /// <param name="derivatives"></param>
        /// <param name="derivativesTotalLenght"></param>
        /// <param name="deltas"></param>
        /// <param name="deltasTotalLenght"></param>
        /// <param name="errorMargins"></param>
        /// <param name="errorMarginsTotalLenght"></param>
        /// <param name="activationUnits"></param>
        /// <param name="activationUnitsTotalLenght"></param>
        public Tuple(double[][][] upperThetas, double upperThetasTotalLenght, double[][][] derivatives, double derivativesTotalLenght, double[][][] deltas, double deltasTotalLenght, double[][] errorMargins, double errorMarginsTotalLenght, double[][][] activationUnits, double activationUnitsTotalLenght)
        {
            this.upperThetas = upperThetas;
            this.upperThetasTotalLenght = upperThetasTotalLenght;
            this.derivatives = derivatives;
            this.derivativesTotalLenght = derivativesTotalLenght;
            this.deltas = deltas;
            this.deltasTotalLenght = deltasTotalLenght;
            this.errorMargins = errorMargins;
            this.errorMarginsTotalLenght = errorMarginsTotalLenght;
            this.activationUnits = activationUnits;
            this.activationUnitsTotalLenght = activationUnitsTotalLenght;
        }
    }
}