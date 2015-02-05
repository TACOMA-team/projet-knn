package fr.inria.tacoma.bft.sensorbelief;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class LinearSensorModelTest {

    private FrameOfDiscernment frame;
    private LinearSensorBeliefModel baseModel;

    @Before
    public void setUp() throws Exception {
        frame = FrameOfDiscernment.newFrame("position", "standing", "sitting", "layingDown");
        baseModel = new LinearSensorBeliefModel(frame);
    }


    @Test
    public void toMassFunctionWithConstantMassFunction_toMassFunction_ReturnsTheRightValue() {
        baseModel.addInterpolationPoint(123.0, frame.toStateSet("sitting"), 0.3);
        baseModel.addInterpolationPoint(123.0, frame.toStateSet("sitting", "standing"), 0.7);
        MassFunction expected = new MassFunctionImpl(frame);
        expected.addToFocal(frame.toStateSet("sitting"), 0.3);
        expected.addToFocal(frame.toStateSet("sitting", "standing"), 0.7);
        MassFunction result = baseModel.toMass(200.0);
        assertEquals(expected, result);
    }

    @Test
    public void toMassFunctionWithTwoInterpolationPoint_toMassFunction_ReturnsTheRightValue() {
        baseModel.addInterpolationPoint(100.0,frame.toStateSet("sitting"),  1.0);
        baseModel.addInterpolationPoint(200.0, frame.toStateSet("sitting", "standing"), 1.0);
        MassFunction expected = new MassFunctionImpl(frame);
        expected.addToFocal(frame.toStateSet("sitting"), 0.5);
        expected.addToFocal(frame.toStateSet("sitting", "standing"), 0.5);
        MassFunction result = baseModel.toMass(150.0);
        assertEquals(expected, result);
    }

    @Test
    public void toMassFunctionWithThreeInterpolationPoint_toMassFunction_ReturnsTheRightValue() {
        SensorBeliefModel model = modelWith3InterpolationPoints();
        MassFunction expected = new MassFunctionImpl(frame);
        expected.addToFocal(frame.toStateSet("sitting", "standing"), 0.8);
        expected.addToFocal(frame.toStateSet("sitting", "standing","layingDown"), 0.2);
        MassFunction result = model.toMass(220.0);
        assertEquals(expected, result);
    }

    @Test
    public void toMassFunction_WithoutSensorMeasure_ReturnsVacuousMassFunction() {
        MassFunction vacuousMass = new MassFunctionImpl(this.frame);
        vacuousMass.putRemainingOnIgnorance();
        assertEquals(vacuousMass, modelWith3InterpolationPoints().toMassWithoutValue());
    }


    private SensorBeliefModel modelWith3InterpolationPoints() {
        baseModel.addInterpolationPoint(100.0, frame.toStateSet("sitting"), 1.0);
        baseModel.addInterpolationPoint(200.0, frame.toStateSet("sitting", "standing"),  1.0);
        baseModel.addInterpolationPoint(300.0,
                frame.toStateSet("sitting", "standing", "layingDown"), 1.0);
        return baseModel;
    }


}
