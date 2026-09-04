import { expect, it } from "vitest";

it("has a browser url", () => {
  expect(document.URL).toBe("http://localhost:5173/");
});
