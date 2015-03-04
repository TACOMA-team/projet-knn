package fr.inria.tacoma.bft.sensorbelief;

import fr.inria.tacoma.bft.combinations.BeliefCombination;
import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;

/**
 * A model which keep the previous generated mass function and use it to
 * compute later conversion from sensor value to mass function.
 *
 * This model contains an underlying model, typically a LinearSensorBeliefModel.
 * When receiving a new sensor data, it uses this underlying model to create a
 * new mass function. The old mass function is weakened according to the time
 * elapsed since the last call to toMass. The model then fuse with a
 * Dubois and Prade combination the old function and the new function to get the
 * new mass function.
 *
 * This is a way of using old evidence in order to create a new belief. If the
 * new sensor value is not specific we can keep the old one (and discount the mass
 * function depending on the age of the evidence).
 */
public class FusionSensorBeliefModel<T> implements TimeDependentSensorBeliefModel<T> {


    private final double maxTime;
    private SensorBeliefModel<T> underlyingModel;
    private MassFunction currentMass;
    private BeliefCombination combination;
    private double oldTimeStamp;

    /**
     * Creates a model with temporization using fusion. It needs an underlying
     * model and a maxTime in seconds. This time is the duration of an evidence.
     * After maxTime seconds, the old evidence will be equal to a mass function
     * with only the full ignorance set as focal point.
     * @param underlyingModel underlying model used to compute new mass functions
     * @param maxTime duration for a mass function
     */
    public FusionSensorBeliefModel(SensorBeliefModel<T> underlyingModel, double maxTime) {
        this.underlyingModel = underlyingModel;
        this.maxTime = maxTime;
        this.combination = Combinations::duboisAndPrade;
    }

    @Override
    public MassFunction toMass(T sensorValue) {
        double newTimeStamp = (double)System.nanoTime() / 1e9;
        return toMass(sensorValue, newTimeStamp - this.oldTimeStamp);
    }

    @Override
    public MassFunction toMass(T sensorValue, double elapsedTime) {
        MassFunction newMass = this.underlyingModel.toMass(sensorValue);
        this.oldTimeStamp = this.oldTimeStamp + elapsedTime;

        if(this.currentMass == null) {
            this.currentMass = newMass;
            return new MassFunctionImpl(currentMass);
        }
        this.currentMass.discount(elapsedTime / this.maxTime);

        this.currentMass = combination.apply(this.currentMass, newMass);

        return new MassFunctionImpl(currentMass);
    }

    @Override
    public MassFunction toMassWithoutValue() {
        double newTimeStamp = (double)System.nanoTime() / 1e9;
        if(this.currentMass == null) {
            this.oldTimeStamp = newTimeStamp;
        }
        return toMassWithoutValue(newTimeStamp - this.oldTimeStamp);
    }

    @Override
    public MassFunction toMassWithoutValue(double elapsedTimeSeconds) {

        if(this.currentMass == null) { //we have no previous evidence to rely on
            MassFunction massFunction = new MassFunctionImpl(this.getFrame());
            massFunction.putRemainingOnIgnorance();
            return massFunction;
        }
        MassFunction weakened = new MassFunctionImpl(this.currentMass);
        weakened.discount(elapsedTimeSeconds / this.maxTime);
        return weakened;
    }

    @Override
    public FrameOfDiscernment getFrame() {
        return underlyingModel.getFrame();
    }

    /**
     * @return the model used by this fusion model.
     */
    public SensorBeliefModel getUnderlyingModel() {
        return underlyingModel;
    }
}
