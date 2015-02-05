package fr.inria.tacoma.bft.util;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class MassUtilTest {

    private FrameOfDiscernment frame;
    private MassFunction sittingMass;
    private MassFunction mass2;
    private MassFunction mass3;

    @Before
    public void setUp() throws Exception {
        frame = FrameOfDiscernment.newFrame("position", "sitting", "standing", "layingDown");

        sittingMass = new MassFunctionImpl(frame);
        sittingMass.set(frame.toStateSet("sitting"), 1.0);

        mass2 = new MassFunctionImpl(frame);
        mass2.set(frame.toStateSet("sitting"), 0.2);
        mass2.set(frame.toStateSet("sitting", "layingDown"), 0.3);
        mass2.putRemainingOnIgnorance();

        mass3 = new MassFunctionImpl(frame);
        mass3.set(frame.toStateSet("standing"), 0.3);
        mass3.putRemainingOnIgnorance();
    }

    @Test
    public void Specificity_forFunctionWithOnlyOneFocalOfCard1_ReturnsOne() {
        assertEquals(1.0, Mass.specificity(sittingMass), MassFunctionImpl.PRECISION);
    }

    @Test
    public void Specificity_forMoreComplexMass_ReturnsTheRightValue() {
        assertEquals(0.2 / 1 + 0.3 / 2 + 0.5 / 3,
                Mass.specificity(mass2), MassFunctionImpl.PRECISION);
    }

    @Test
    public void scalarProduct_givesExpectedResult() {
        assertEquals(0.7066666666666667, Mass.scalarProduct(mass2, mass2),
                     MassFunctionImpl.PRECISION);
    }

    @Test
    public void scalarProduct_givesExpectedResult2() {
        assertEquals(0.5166666666666666, Mass.scalarProduct(sittingMass, mass2),
                     MassFunctionImpl.PRECISION);
    }
    @Test
     public void scalarProduct_givesExpectedResult3() {
        assertEquals(0.72, Mass.scalarProduct(mass3, mass3), MassFunctionImpl.PRECISION);
    }

    @Test
    public void scalarProduct_withCompletelyConflictingMasses_givesExpectedResult3() {
        MassFunction layingDownMass = new MassFunctionImpl(frame);
        layingDownMass.set(frame.toStateSet("standing"), 1.0);
        assertEquals(0, Mass.scalarProduct(sittingMass, layingDownMass), MassFunctionImpl.PRECISION);
    }

    @Test
    public void scalarProduct_isSymetric() {
        assertEquals(Mass.scalarProduct(mass2, mass3), Mass.scalarProduct(mass3, mass2),
                     MassFunctionImpl.PRECISION);
    }

    @Test
    public void jousselmeDistance_withTheSameMass_Returns0() {
        assertEquals(0.0, Mass.jousselmeDistance(sittingMass,sittingMass),
                     MassFunctionImpl.PRECISION);
        assertEquals(0.0, Mass.jousselmeDistance(mass2,mass2), MassFunctionImpl.PRECISION);
        assertEquals(0.0, Mass.jousselmeDistance(mass3,mass3), MassFunctionImpl.PRECISION);
    }

    @Test
    public void jousselmeDistance_withDifferentMass_ReturnsExpectedResult() {
        assertEquals(0.35590260840104393, Mass.jousselmeDistance(mass2, mass3),
                     MassFunctionImpl.PRECISION);
    }

    @Test
    public void jousselmeDistance_isSymmetric() {
        assertEquals(Mass.jousselmeDistance(mass2, mass3), Mass.jousselmeDistance(mass3, mass2),
                     MassFunctionImpl.PRECISION);
        assertEquals(Mass.jousselmeDistance(mass2, sittingMass),
                     Mass.jousselmeDistance(sittingMass, mass2),
                     MassFunctionImpl.PRECISION);
    }

    /*
     * consonant functions:
     */


    private MassFunctionImpl createNonConsonantMass() {
        MassFunctionImpl massFunction = new MassFunctionImpl(frame);
        massFunction.set(frame.toStateSet("sitting"), 0.6);
        massFunction.set(frame.toStateSet("standing"), 0.4);
        return massFunction;
    }

    @Test
    public void isConsonant_withConsonantFunction_ReturnsTrue() {
        assertTrue(Mass.isConsonant(mass2));
    }
    @Test
     public void isConsonant_withNonConsonantFunction_ReturnsFalse() {
        MassFunctionImpl massFunction = createNonConsonantMass();
        assertFalse(Mass.isConsonant(massFunction));
    }

    @Test
    public void toConsonant_withConsonantFunction_ReturnsTheSameFunction() {
        assertEquals(mass2, Mass.toConsonant(mass2));
    }

    @Test
    public void toConsonant_withNonConsonantMass_ReturnsConsonantMass() {
        MassFunctionImpl massFunction = createNonConsonantMass();
        MassFunction shouldBeConsonant = Mass.toConsonant(massFunction);
        assertTrue(shouldBeConsonant + " is not consonant.", Mass.isConsonant(shouldBeConsonant));
    }
    @Test
    public void toConsonant_withNonConsonantMass_ReturnsExpectedResult() {
        MassFunctionImpl massFunction = createNonConsonantMass();
        MassFunction consonantMass = Mass.toConsonant(massFunction);
        MassFunction expected = new MassFunctionImpl(frame);
        expected.set(frame.toStateSet("sitting"), 0.2);
        expected.set(frame.toStateSet("sitting", "standing"), 0.8);
        assertEquals(expected, consonantMass);
    }
}
