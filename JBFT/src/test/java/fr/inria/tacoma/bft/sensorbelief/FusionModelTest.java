package fr.inria.tacoma.bft.sensorbelief;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class FusionModelTest {
    private FrameOfDiscernment frame;
    private LinearSensorBeliefModel baseModel;
    private FusionSensorBeliefModel<Double> fusionModel;

    @Before
    public void setUp() {
        frame = FrameOfDiscernment.newFrame("position", "standing", "sitting", "layingDown");
        baseModel = new LinearSensorBeliefModel(frame);
        baseModel.addInterpolationPoint(100.0, frame.toStateSet("sitting"), 1.0);
        baseModel.addInterpolationPoint(200.0, frame.toStateSet("sitting", "standing"), 1.0);
        baseModel.addInterpolationPoint(300.0, frame.fullIgnoranceSet(), 1.0);
        fusionModel = new FusionSensorBeliefModel<>(baseModel, 1.0);
    }

    @Test
    public void sensorModelWithFusion_firstCalltoMassFunction_ReturnsTheSameAsUnderlyingModel() {
        assertEquals(baseModel.toMass(100.0), fusionModel.toMass(100.0, 0));
    }

    @Test
    public void secondCalltoMassFunction_withNoTimeBetweenEvidences_isEquivalentToDuboisAndPradeCombination() {
        fusionModel.toMass(100.0, 0); //first call
        MassFunction expected = Combinations.duboisAndPrade(
                baseModel.toMass(100.0), baseModel.toMass(200.0));
        assertEquals(expected,
                fusionModel.toMass(200.0, 0));
    }

    @Test
    public void secondCalltoMassFunction_withTimeBetweenEvidences_ReturnsRightResult() {
        fusionModel.toMass(100.0, 0); //first call
        MassFunction expected = new MassFunctionImpl(this.frame);
        expected.addToFocal(this.frame.toStateSet("sitting"), 0.5);
        expected.addToFocal(this.frame.toStateSet("sitting", "standing"), 0.5);
        assertEquals(expected, this.fusionModel.toMass(200.0, 0.5));
    }

    @Test
    public void toMassFunction_WithoutSensorMeasureAsFirstCall_ReturnsVacuousMassFunction() {
        MassFunction vacuousMass = new MassFunctionImpl(this.frame);
        vacuousMass.putRemainingOnIgnorance();
        assertEquals(vacuousMass, fusionModel.toMassWithoutValue());
    }

    @Test
    public void toMassFunction_WithoutSensorMeasureAsSecond_ReturnsWeakeanedPreviousFunction() {
        MassFunction firstMass = this.fusionModel.toMass(200.0);
        firstMass.discount(0.5);
        assertEquals(firstMass, this.fusionModel.toMassWithoutValue(0.5));
    }
}
