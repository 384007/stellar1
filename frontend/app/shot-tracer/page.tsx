import { redirect } from "next/navigation";

export default function ShotTracerEntryPage() {
  redirect("/shot-lab?tab=trajectory");
}
