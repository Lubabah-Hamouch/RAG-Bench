import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Meter } from "../components/ui";

const shown = () => screen.getByTestId("meter-value").textContent;

// Regression: a live card renders unscored (null) first, then receives its score. A score of exactly
// 0 animated from 0 to 0, fired no change event, and kept showing "-" instead of "0.00".
describe("Meter", () => {
  it("shows 0.00 when a score of exactly zero arrives after an unscored render", async () => {
    const { rerender } = render(<Meter label="Coverage" value={null} color="red" />);
    expect(shown()).toBe("–");
    rerender(<Meter label="Coverage" value={0} color="red" />);
    await waitFor(() => expect(shown()).toBe("0.00"));
  });

  it("settles on the exact score after animating", async () => {
    const { rerender } = render(<Meter label="Grounded" value={null} color="red" />);
    rerender(<Meter label="Grounded" value={0.75} color="red" />);
    await waitFor(() => expect(shown()).toBe("0.75"));
    rerender(<Meter label="Grounded" value={1} color="red" />);
    await waitFor(() => expect(shown()).toBe("1.00"));
    rerender(<Meter label="Grounded" value={0} color="red" />);
    await waitFor(() => expect(shown()).toBe("0.00"));
  });

  it("shows a dash only when there is no score", () => {
    render(<Meter label="FCS" value={undefined} color="red" />);
    expect(shown()).toBe("–");
  });

  it("shows a score that is present from the first render", () => {
    render(<Meter label="FCS" value={0} color="red" />);
    expect(shown()).toBe("0.00");
  });
});
