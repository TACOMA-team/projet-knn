package fr.inria.tacoma.bft.core;


import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

public class MassFunctionTest {

	private MassFunction massFunction;
	private FrameOfDiscernment frame;

	@Before
    public void setup() {
		frame = FrameOfDiscernment.newFrame("position", "standing", "sitting", "layingDown");
		massFunction = new MassFunctionImpl(frame);
        massFunction.addToFocal(frame.toStateSet("sitting"), 0.2);
        massFunction.addToFocal(frame.toStateSet("sitting", "layingDown"), 0.3);
	}

    @Test
    public void MassFunction_Copy_ResultIsEqualsToOriginal() {
        MassFunction mass = getMassWithTotalIgnorance();
        assertEquals(mass, new MassFunctionImpl(mass));
    }

	@Test(expected = IllegalArgumentException.class)
	public void addFocal_NegativeNumberAsValue_ThrowsException() {
		massFunction.addToFocal(frame.toStateSet("standing"), -.025);
	}

	@Test(expected = ArithmeticException.class)
	public void normalize_WithEmptyFunction_ThrowsException() {
		new MassFunctionImpl(frame).normalize();
	}

	@Test(expected = ArithmeticException.class)
	public void putRemainingOnIgnorance_WithMassGreaterThan1_ThrowsException() {
		massFunction.addToFocal(frame.toStateSet("standing"), 1.2);
        massFunction.putRemainingOnIgnorance();
	}

	@Test
	public void MassFunctionGenerator_getTotalAssignedMass_ReturnsTheRightNumber() {
		assertEquals(0.5, massFunction.getTotalAssignedMass(), 1e-12);
	}

	@Test
	 public void getTotalAssignedMass_afterNormalizingTheFunction_ReturnsOne() {
		assertEquals(1.0, getNormalizedMassFunction().getTotalAssignedMass(), 1e-12);
	}

    @Test
    public void MassFunction_setValue_UpdateTotalAssignedMass() {
        assertEquals(0.5, this.massFunction.getTotalAssignedMass(), 1e-12);
        this.massFunction.set(this.frame.toStateSet("sitting"), 0.5);
        assertEquals(0.8, this.massFunction.getTotalAssignedMass(), 1e-12);
    }

	/*
	 * Tests for a function with remaining mass put on total ignorance :
	 */

	@Test
	public void putRemainingOnIgnorance_withAssignedValueLowerThan1_DoesNotChangeValueForOtherElements() {
		assertMassEquals(getMassWithTotalIgnorance(), 0.2, "sitting");
	}

	@Test
	public void putRemainingOnIgnorance_withAssignedValueLowerThan1_DoesNotChangeValueForOtherElements2() {
		assertMassEquals(getMassWithTotalIgnorance(), 0.3, "sitting", "layingDown");
	}

	@Test
	public void MassFunctionGenerator_putRemainingMassOnIgnorance_AssignsRightValueTotalIgnorance() {
		assertMassEquals(getMassWithTotalIgnorance(), 0.5, "sitting", "layingDown", "standing");
	}

	@Test(expected = IllegalArgumentException.class)
	public void MassFunction_gettingNonExistingElement_ThrowsException() {
		getMassWithTotalIgnorance().get("nonExisting");
	}


    @Test
    public void MassFunction_gettingNonFocalPoint_Returns0() {
        assertMassEquals(getMassWithTotalIgnorance(), 0.0, "standing");
    }

	@Test
	public void getValue_withNonFocalElement_Returns0() {
		assertMassEquals(getMassWithTotalIgnorance(), 0, "standing");
	}


	@Test
	public void putRemainingMassOnIgnorance_WithValueAlreadySetForTotalIgnorance_AssignsRightValueToTotalIgnorance() {
		massFunction.addToFocal(frame.toStateSet("sitting", "standing", "layingDown"), 0.2);
        massFunction.putRemainingOnIgnorance();
		assertMassEquals(massFunction, 0.5, "sitting", "standing", "layingDown");
	}

	@Test
	public void addValueToFocal_withFocalAlreadyExisting_SumsNewAndFormerValue() {
		this.massFunction.addToFocal(frame.toStateSet("sitting"), 0.2);
        massFunction.putRemainingOnIgnorance();
		assertMassEquals(massFunction, 0.4, "sitting");
	}

	/*
	 * Tests for a normalized mass :
	 */

	@Test
	public void normalize_withAssignedValueLowerThan1_AssignsRightValueToElements() {
		assertMassEquals(getNormalizedMassFunction(), 0.4, "sitting");
	}

	@Test
	public void normalize_withAssignedValueLowerThan1_AssignsRightValueToElements2() {
		assertMassEquals(getNormalizedMassFunction(), 0.6, "sitting", "layingDown");
	}

    /*
     * Tests for weakening
     */

    @Test
    public void weaken_byTenPercent_GivesRightMassFunction() {
        MassFunction result = getNormalizedMassFunction();
        result.discount(0.1);
        MassFunction expected = new MassFunctionImpl(frame);
        expected.set(frame.toStateSet("sitting"), 0.36);
        expected.set(frame.toStateSet("sitting", "layingDown"), 0.54);
        expected.set(frame.fullIgnoranceSet(), 0.1);
        assertEquals(expected, result);
    }

	private MassFunction getNormalizedMassFunction() {
        MassFunction normalized = new MassFunctionImpl(massFunction);
        normalized.normalize();
		return normalized;
	}

	private MassFunction getMassWithTotalIgnorance() {
        MassFunction mass = new MassFunctionImpl(massFunction);
        mass.putRemainingOnIgnorance();
		return mass;
	}

	private void assertMassEquals(MassFunction massFunction, double expected, String... elements) {
		double precision = 1e-12;
		assertEquals(expected, massFunction.get(elements), precision);
	}

    /*
     * toString
     */
    @Test
    public void MassFunction_toString_returnValidJsonString(){
        String content = getMassWithTotalIgnorance().toString();
        try {
            new ObjectMapper().readTree(content);
        } catch (IOException e) {
            fail("Not a valid json string: " + content );
        }
    }
}
